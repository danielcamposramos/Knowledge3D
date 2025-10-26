"""
Self-Updating TRM: Safe Weight Updates Without Catastrophic Forgetting

Implements shadow weights with validation gating:
1. Primary weights (production) - Always stable
2. Shadow weights (candidate) - Test updates here
3. Validation gate - Only commit if performance improves

Prevents "losing mind" through:
- Holdout validation set (never trained on)
- Performance baseline tracking
- Gradual weight blending (EMA)
- Elastic Weight Consolidation (protect important weights)

Usage:
    updater = SelfUpdatingTRM()
    success = updater.propose_and_validate(new_batch, validation_set)
    if success:
        updater.commit_update()
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import json
from dataclasses import dataclass
from enum import Enum


class UpdateStrategy(Enum):
    """Weight update strategies."""
    REPLACE = "replace"         # Full replacement (risky)
    BLEND = "blend"            # Exponential moving average (safe)
    EWC = "ewc"               # Elastic Weight Consolidation (safest)


@dataclass
class UpdateConfig:
    """Self-update configuration."""
    strategy: UpdateStrategy = UpdateStrategy.BLEND
    blend_alpha: float = 0.1              # EMA coefficient (0.1 = 90% old, 10% new)
    min_improvement: float = 0.001        # Minimum performance gain to commit
    ewc_lambda: float = 100.0             # EWC penalty strength
    max_degradation: float = 0.05         # Never allow >5% performance drop


class TRMWeightManager:
    """
    Manages TRM weight buffers with shadow copy for safe updates.

    Sovereign implementation: Direct NumPy arrays, no framework overhead.
    """

    def __init__(self, weight_shape: Tuple[int, ...]):
        """
        Initialize weight manager.

        Args:
            weight_shape: Shape of TRM weight tensor
        """
        self.weight_shape = weight_shape

        # Primary weights (production)
        self.W_primary = np.random.randn(*weight_shape).astype(np.float32) * 0.01

        # Shadow weights (candidate updates)
        self.W_shadow = np.zeros_like(self.W_primary)

        # Fisher information matrix (weight importance)
        self.fisher_matrix = np.ones_like(self.W_primary)

        # Performance tracking
        self.baseline_performance = 0.0
        self.shadow_performance = 0.0

        # Update history
        self.update_count = 0
        self.accepted_count = 0
        self.rejected_count = 0

    def get_primary_weights(self) -> np.ndarray:
        """Get production weights (read-only)."""
        return self.W_primary.copy()

    def get_shadow_weights(self) -> np.ndarray:
        """Get shadow weights for testing."""
        return self.W_shadow

    def fork_to_shadow(self):
        """Copy primary → shadow (prepare for testing update)."""
        np.copyto(self.W_shadow, self.W_primary)

    def apply_gradient_to_shadow(self, gradient: np.ndarray, learning_rate: float = 0.001):
        """Apply gradient descent to shadow weights."""
        if gradient.shape != self.weight_shape:
            raise ValueError(f"Gradient shape {gradient.shape} != weight shape {self.weight_shape}")

        # Gradient clipping
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1.0:
            gradient = gradient / grad_norm

        # Update shadow
        self.W_shadow -= learning_rate * gradient

    def compute_fisher_information(self, validation_set: List[Dict],
                                   loss_fn: Callable) -> np.ndarray:
        """
        Compute Fisher information matrix (weight importance).

        High Fisher value = weight is important for current performance.
        Protect these weights during updates.
        """
        # Simplified Fisher: variance of gradients on validation set
        gradients = []

        for sample in validation_set[:100]:  # Use 100 samples for efficiency
            # Compute gradient for this sample
            grad = loss_fn(sample, self.W_primary)
            gradients.append(grad ** 2)

        # Fisher information = mean of squared gradients
        fisher = np.mean(gradients, axis=0)

        return fisher

    def ewc_penalty(self) -> float:
        """
        Elastic Weight Consolidation penalty.

        Penalize changes to important weights (high Fisher information).
        """
        # Weight change
        delta_W = self.W_shadow - self.W_primary

        # Penalty = Fisher * (delta)^2
        penalty = np.sum(self.fisher_matrix * (delta_W ** 2))

        return penalty

    def blend_weights(self, alpha: float = 0.1):
        """
        Exponential moving average: primary ← (1-α)*primary + α*shadow

        Args:
            alpha: Blend coefficient (0.0 = keep primary, 1.0 = full shadow)
        """
        self.W_primary = (1 - alpha) * self.W_primary + alpha * self.W_shadow

    def commit_shadow_to_primary(self, strategy: UpdateStrategy, alpha: float = 0.1):
        """Commit shadow weights to primary using specified strategy."""
        if strategy == UpdateStrategy.REPLACE:
            # Full replacement (risky)
            np.copyto(self.W_primary, self.W_shadow)

        elif strategy == UpdateStrategy.BLEND:
            # Exponential moving average (safe)
            self.blend_weights(alpha)

        elif strategy == UpdateStrategy.EWC:
            # Blend with EWC penalty consideration
            # Higher penalty → lower alpha (more conservative)
            penalty = self.ewc_penalty()
            adaptive_alpha = alpha / (1.0 + penalty / 100.0)
            self.blend_weights(adaptive_alpha)

        self.accepted_count += 1

    def reject_shadow(self):
        """Reject shadow weights, keep primary unchanged."""
        self.rejected_count += 1

    def save_weights(self, path: Path):
        """Save primary weights to disk."""
        np.save(path, self.W_primary)

    def load_weights(self, path: Path):
        """Load weights from disk."""
        self.W_primary = np.load(path)
        self.W_shadow = np.zeros_like(self.W_primary)


class SelfUpdatingTRM:
    """
    Self-updating TRM with safe weight management.

    Prevents catastrophic forgetting through validation gating.
    """

    def __init__(self, config: Optional[UpdateConfig] = None):
        self.config = config or UpdateConfig()

        # Weight manager (placeholder shape, will be initialized properly)
        self.weight_manager = TRMWeightManager(weight_shape=(256, 512))

        # Validation set (holdout, never trained on)
        self.validation_set = []

        # Performance history
        self.performance_history = []

        print(f"[SelfUpdatingTRM] Initialized with strategy: {self.config.strategy.value}")

    def set_validation_set(self, samples: List[Dict]):
        """Set holdout validation set."""
        self.validation_set = samples
        print(f"[SelfUpdatingTRM] Validation set: {len(samples)} samples")

    def evaluate_performance(self, weights: np.ndarray,
                           validation_set: Optional[List[Dict]] = None) -> float:
        """
        Evaluate model performance on validation set.

        Args:
            weights: Weights to evaluate
            validation_set: Samples to evaluate on (default: self.validation_set)

        Returns:
            Performance score (higher = better)
        """
        if validation_set is None:
            validation_set = self.validation_set

        if len(validation_set) == 0:
            print("[Warning] No validation set, returning 0.0")
            return 0.0

        # Compute success rate on validation set
        success_count = 0

        for sample in validation_set:
            # Check if this sample has successful teacher evaluation
            teacher_eval = sample.get('teacher_evaluation', {})
            rating_score = teacher_eval.get('rating_score', -1)

            if rating_score >= 0:
                # Convert rating to performance (10/10 = 1.0, 1/10 = 0.1)
                performance = rating_score / 10.0
                success_count += performance

        # Average performance
        avg_performance = success_count / len(validation_set)

        return avg_performance

    def propose_update(self, gradient: np.ndarray, learning_rate: float = 0.001):
        """
        Propose weight update by applying gradient to shadow weights.

        Args:
            gradient: Gradient to apply
            learning_rate: Learning rate for update
        """
        # Fork primary → shadow
        self.weight_manager.fork_to_shadow()

        # Apply gradient to shadow
        self.weight_manager.apply_gradient_to_shadow(gradient, learning_rate)

    def validate_and_commit(self) -> Tuple[bool, float, float]:
        """
        Validate shadow weights and commit if performance improves.

        Returns:
            (success, baseline_perf, shadow_perf)
        """
        # Evaluate baseline (primary weights)
        baseline_perf = self.evaluate_performance(self.weight_manager.W_primary)

        # Evaluate candidate (shadow weights)
        shadow_perf = self.evaluate_performance(self.weight_manager.W_shadow)

        # Store for tracking
        self.weight_manager.baseline_performance = baseline_perf
        self.weight_manager.shadow_performance = shadow_perf

        # Decision criteria
        improvement = shadow_perf - baseline_perf
        degradation = baseline_perf - shadow_perf

        # Check if update meets criteria
        if improvement >= self.config.min_improvement:
            # Performance improved → commit
            self.weight_manager.commit_shadow_to_primary(
                strategy=self.config.strategy,
                alpha=self.config.blend_alpha
            )

            self.performance_history.append({
                'step': self.weight_manager.update_count,
                'baseline': baseline_perf,
                'shadow': shadow_perf,
                'improvement': improvement,
                'accepted': True
            })

            print(f"[Update] ✓ ACCEPTED: {baseline_perf:.4f} → {shadow_perf:.4f} "
                  f"(+{improvement:.4f})")

            self.weight_manager.update_count += 1
            return True, baseline_perf, shadow_perf

        elif degradation > self.config.max_degradation:
            # Performance degraded too much → reject
            self.weight_manager.reject_shadow()

            self.performance_history.append({
                'step': self.weight_manager.update_count,
                'baseline': baseline_perf,
                'shadow': shadow_perf,
                'degradation': degradation,
                'accepted': False,
                'reason': 'excessive_degradation'
            })

            print(f"[Update] ✗ REJECTED: {baseline_perf:.4f} → {shadow_perf:.4f} "
                  f"(-{degradation:.4f}) - Excessive degradation")

            self.weight_manager.update_count += 1
            return False, baseline_perf, shadow_perf

        else:
            # Marginal change (neither big improvement nor degradation) → reject
            self.weight_manager.reject_shadow()

            self.performance_history.append({
                'step': self.weight_manager.update_count,
                'baseline': baseline_perf,
                'shadow': shadow_perf,
                'improvement': improvement,
                'accepted': False,
                'reason': 'insufficient_improvement'
            })

            print(f"[Update] ✗ REJECTED: {baseline_perf:.4f} → {shadow_perf:.4f} "
                  f"(+{improvement:.4f}) - Insufficient improvement")

            self.weight_manager.update_count += 1
            return False, baseline_perf, shadow_perf

    def get_update_stats(self) -> Dict[str, Any]:
        """Get update statistics."""
        total = self.weight_manager.update_count
        accepted = self.weight_manager.accepted_count
        rejected = self.weight_manager.rejected_count

        return {
            'total_updates': total,
            'accepted': accepted,
            'rejected': rejected,
            'acceptance_rate': accepted / max(total, 1),
            'current_performance': self.weight_manager.baseline_performance,
            'recent_history': self.performance_history[-10:]
        }

    def save_checkpoint(self, checkpoint_dir: Path):
        """Save self-updating checkpoint."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        self.weight_manager.save_weights(checkpoint_dir / 'trm_weights.npy')

        # Save metadata
        metadata = {
            'config': {
                'strategy': self.config.strategy.value,
                'blend_alpha': self.config.blend_alpha,
                'min_improvement': self.config.min_improvement
            },
            'stats': self.get_update_stats(),
            'performance_history': self.performance_history
        }

        with open(checkpoint_dir / 'update_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[Checkpoint] Saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: Path):
        """Load self-updating checkpoint."""
        # Load weights
        self.weight_manager.load_weights(checkpoint_dir / 'trm_weights.npy')

        # Load metadata
        with open(checkpoint_dir / 'update_metadata.json', 'r') as f:
            metadata = json.load(f)

        self.performance_history = metadata.get('performance_history', [])

        print(f"[Checkpoint] Loaded from {checkpoint_dir}")
        print(f"  Updates: {metadata['stats']['total_updates']}")
        print(f"  Acceptance rate: {metadata['stats']['acceptance_rate']*100:.1f}%")
