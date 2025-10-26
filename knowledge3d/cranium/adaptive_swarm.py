"""
Adaptive Swarm TRM: Self-Updating Multi-Specialist System

Integrates Matryoshka base with self-updating specialists for continual learning.

Architecture:
    1. Self-updating base model (shared by all specialists)
    2. Self-updating specialist adapters (independent evolution)
    3. Dynamic dimension selection (task complexity → capacity)
    4. Safe weight updates (shadow weights + validation gates)

Key Features:
- Base model improvements benefit ALL specialists automatically
- Each specialist evolves independently on its task domain
- Automatic dimension scaling (64 dims → 16K dims as needed)
- No catastrophic forgetting (validation gating)
- Memory efficient (18× reduction vs full specialists)

Usage:
    # Create swarm
    swarm = AdaptiveSwarmTRM(base_dims=2048)

    # Register specialists
    swarm.register_specialist('ocr', required_dims=512, rank=32)
    swarm.register_specialist('math', required_dims=1024, rank=64)
    swarm.register_specialist('code', required_dims=2048, rank=128)

    # Train base on general reasoning
    swarm.train_base_epoch(general_samples, validation_samples)

    # Train specialist on domain-specific data
    swarm.train_specialist_epoch('ocr', ocr_samples, ocr_validation)

    # Inference with automatic specialist selection
    output = swarm.forward(input_data, specialist='auto')
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
from dataclasses import dataclass
from datetime import datetime

from knowledge3d.cranium.matryoshka_trm import MatryoshkaTRM, DimensionSelector
from knowledge3d.cranium.trm_adapters import (
    SelfUpdatingAdapter,
    AdapterConfig,
    _to_serializable,
)


@dataclass
class SwarmConfig:
    """Configuration for adaptive swarm."""
    base_dims: int = 2048                    # Base model dimension
    min_dims: int = 64                       # Minimum dimension
    base_learning_rate: float = 0.001        # Base model LR
    specialist_learning_rate: float = 0.002  # Specialist LR (can be higher)
    validation_split: float = 0.1            # Validation holdout
    checkpoint_interval: int = 100           # Save every N steps
    enable_auto_expansion: bool = True       # Auto-expand dims when needed
    expansion_threshold: float = 0.95        # Expand if complexity > threshold


class AdaptiveSwarmTRM:
    """
    Adaptive swarm with self-updating base and specialists.

    Combines:
    - MatryoshkaTRM (variable dimensionality base)
    - SelfUpdatingAdapter (per-specialist adapters)
    - Validation gating (safe updates)
    - Dynamic dimension selection
    """

    def __init__(self, config: Optional[SwarmConfig] = None):
        """
        Initialize adaptive swarm.

        Args:
            config: Swarm configuration
        """
        self.config = config or SwarmConfig()

        # Matryoshka base (variable dimensionality)
        self.base = MatryoshkaTRM(
            max_dims=self.config.base_dims,
            min_dims=self.config.min_dims
        )

        # Training state
        self.base_step = 0
        self.specialist_steps: Dict[str, int] = {}

        # Validation sets (per specialist + general)
        self.base_validation_samples = []

        # Base self-updating mechanism
        self.W_base_shadow = np.zeros_like(self.base.W_base_full)
        self.base_baseline_performance = 0.0
        self.base_update_count = 0
        self.base_accepted_count = 0
        self.base_rejected_count = 0

        print(f"[AdaptiveSwarmTRM] Initialized")
        print(f"  Base dimensions: {self.config.base_dims}")
        print(f"  Dimension range: {self.config.min_dims} - {self.config.base_dims}")
        print(f"  Auto-expansion: {'Enabled' if self.config.enable_auto_expansion else 'Disabled'}")

    def register_specialist(self, name: str, required_dims: Optional[int] = None,
                           rank: Optional[int] = None,
                           adapter_config: Optional[AdapterConfig] = None):
        """
        Register new specialist.

        Args:
            name: Specialist identifier
            required_dims: Dimension requirement (None = auto-select)
            rank: Adapter rank
            adapter_config: Adapter configuration
        """
        # Auto-select dimensions if not specified
        if required_dims is None:
            # Default: Use half of base dims (good starting point)
            required_dims = self.config.base_dims // 2

        # Register with base
        self.base.register_specialist(name, required_dims, rank, adapter_config)

        # Initialize specialist training state
        self.specialist_steps[name] = 0

        print(f"[AdaptiveSwarmTRM] Specialist '{name}' registered")

    def set_base_validation_samples(self, samples: List[Dict]):
        """Set validation samples for base model."""
        self.base_validation_samples = samples
        print(f"[AdaptiveSwarmTRM] Base validation set: {len(samples)} samples")

    def set_specialist_validation_samples(self, specialist_name: str, samples: List[Dict]):
        """Set validation samples for specialist."""
        if specialist_name not in self.base.specialists:
            raise ValueError(f"Unknown specialist: {specialist_name}")

        adapter = self.base.specialists[specialist_name]['adapter']
        adapter.set_validation_samples(samples)

    def train_base_epoch(self, train_samples: List[Dict],
                        eval_fn: Callable[[np.ndarray, List], float],
                        use_self_update: bool = True) -> Dict[str, float]:
        """
        Train base model for one epoch.

        Args:
            train_samples: Training samples
            eval_fn: Evaluation function (weights, samples) → performance
            use_self_update: If True, use shadow weights + validation

        Returns:
            Training statistics
        """
        print(f"\n[Base Training] Starting epoch over {len(train_samples)} samples")

        total_loss = 0.0

        for i, sample in enumerate(train_samples):
            # Compute gradient (placeholder - real implementation would compute actual gradient)
            # In practice, this would be backprop through TRM
            gradient = self._compute_base_gradient(sample)

            if use_self_update:
                # Apply to shadow
                if i == 0:
                    # Fork to shadow at start
                    np.copyto(self.W_base_shadow, self.base.W_base_full)

                # Update shadow
                self.W_base_shadow -= self.config.base_learning_rate * gradient
            else:
                # Direct update (no validation)
                self.base.W_base_full -= self.config.base_learning_rate * gradient

            # Track loss
            loss = np.linalg.norm(gradient)
            total_loss += loss

            self.base_step += 1

            if (i + 1) % 100 == 0:
                avg_loss = total_loss / (i + 1)
                print(f"  Step {i+1}/{len(train_samples)}: Loss {avg_loss:.4f}")

        avg_loss = total_loss / len(train_samples)

        # Validate and commit if using self-update
        if use_self_update and len(self.base_validation_samples) > 0:
            success = self._validate_and_commit_base(eval_fn)

            return {
                'avg_loss': avg_loss,
                'update_accepted': success,
                'steps': self.base_step
            }

        return {'avg_loss': avg_loss, 'steps': self.base_step}

    def train_specialist_epoch(self, specialist_name: str,
                              train_samples: List[Dict],
                              eval_fn: Callable[[np.ndarray, List], float],
                              use_self_update: bool = True) -> Dict[str, float]:
        """
        Train specialist for one epoch.

        Args:
            specialist_name: Which specialist to train
            train_samples: Training samples
            eval_fn: Evaluation function
            use_self_update: If True, use shadow weights + validation

        Returns:
            Training statistics
        """
        if specialist_name not in self.base.specialists:
            raise ValueError(f"Unknown specialist: {specialist_name}")

        specialist = self.base.specialists[specialist_name]
        adapter = specialist['adapter']
        dims = specialist['dims']

        print(f"\n[Specialist '{specialist_name}'] Training epoch over {len(train_samples)} samples")

        total_loss = 0.0

        for i, sample in enumerate(train_samples):
            # Compute gradient for specialist
            gradient = self._compute_specialist_gradient(specialist_name, sample)

            if use_self_update:
                # Fork to shadow at start
                if i == 0:
                    adapter.fork_to_shadow()

                # Apply to shadow
                adapter.apply_gradient_to_shadow(gradient, lr=self.config.specialist_learning_rate)
            else:
                # Direct update
                adapter.apply_gradient(gradient, lr=self.config.specialist_learning_rate)

            # Track loss
            loss = np.linalg.norm(gradient)
            total_loss += loss

            self.specialist_steps[specialist_name] += 1

            if (i + 1) % 100 == 0:
                avg_loss = total_loss / (i + 1)
                print(f"  Step {i+1}/{len(train_samples)}: Loss {avg_loss:.4f}")

        avg_loss = total_loss / len(train_samples)

        # Validate and commit if using self-update
        if use_self_update and len(adapter.validation_samples) > 0:
            # Get current base weights at specialist's dimension
            base_weights = self.base.get_base_at_dim(dims)

            # Validate and commit
            success, baseline, shadow = adapter.validate_and_commit(base_weights, eval_fn)

            return {
                'avg_loss': avg_loss,
                'update_accepted': success,
                'baseline_performance': baseline,
                'shadow_performance': shadow,
                'steps': self.specialist_steps[specialist_name]
            }

        return {'avg_loss': avg_loss, 'steps': self.specialist_steps[specialist_name]}

    def forward(self, input_data: np.ndarray,
                specialist: Optional[str] = None,
                complexity: Optional[float] = None) -> np.ndarray:
        """
        Forward pass through swarm.

        Args:
            input_data: Input vector
            specialist: Specialist to use (None = base only, 'auto' = auto-select)
            complexity: Task complexity [0-1] for auto-dimension selection

        Returns:
            Output vector
        """
        if specialist is None:
            # Base model only
            if complexity is not None:
                # Select appropriate dimension
                dim = DimensionSelector.select_dim(complexity)
                W = self.base.get_base_at_dim(dim)
            else:
                # Use full base
                W = self.base.W_base_full

            # Resize input to match
            input_resized = self.base._resize_input(input_data, W.shape[0])
            return W @ input_resized

        elif specialist == 'auto':
            # Auto-select specialist based on input
            # For now, use base model (MoE router will enhance this)
            return self.forward(input_data, specialist=None, complexity=complexity)

        else:
            # Use specific specialist
            return self.base.compute_with_specialist(input_data, specialist)

    def forward_moe(self, input_data: np.ndarray,
                   specialist_weights: Dict[str, float]) -> np.ndarray:
        """
        Forward pass with MoE (mixture of specialists).

        Args:
            input_data: Input vector
            specialist_weights: Dict mapping specialist_name → weight

        Returns:
            Blended output
        """
        return self.base.compute_with_moe(input_data, specialist_weights)

    def compute_with_specialist(self, input_data: np.ndarray, specialist_name: str) -> np.ndarray:
        """
        Compute using specific specialist.

        Delegates to MatryoshkaTRM.

        Args:
            input_data: Input vector
            specialist_name: Which specialist to use

        Returns:
            Specialist output
        """
        return self.base.compute_with_specialist(input_data, specialist_name)

    def expand_capacity(self, new_max_dims: int):
        """
        Expand base capacity to new dimension.

        Useful when encountering tasks that exceed current capacity.

        Args:
            new_max_dims: New maximum dimension
        """
        self.base.expand_base_dimensions(new_max_dims)

        # Update shadow weights to match
        self.W_base_shadow = np.zeros_like(self.base.W_base_full)

        print(f"[AdaptiveSwarmTRM] Capacity expanded to {new_max_dims} dims")

    def get_system_stats(self) -> Dict[str, Any]:
        """Get complete system statistics."""
        base_stats = self.base.get_system_stats()

        return {
            'base_model': base_stats['base_model'],
            'base_training': {
                'steps': self.base_step,
                'updates_proposed': self.base_update_count,
                'accepted': self.base_accepted_count,
                'rejected': self.base_rejected_count,
                'acceptance_rate': self.base_accepted_count / max(self.base_update_count, 1),
                'baseline_performance': self.base_baseline_performance
            },
            'specialists': base_stats['specialists'],
            'specialist_training': {
                name: {'steps': self.specialist_steps.get(name, 0)}
                for name in self.base.specialists.keys()
            },
            'total_specialist_params': base_stats['total_specialist_params'],
            'total_params': base_stats['total_params'],
            'total_memory_mb': base_stats['total_memory_mb']
        }

    def save_checkpoint(self, checkpoint_dir: Path):
        """Save complete swarm checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save base system
        self.base.save_all(checkpoint_dir)

        # Save swarm-specific state
        swarm_state = {
            'config': {
                'base_dims': self.config.base_dims,
                'min_dims': self.config.min_dims,
                'base_learning_rate': self.config.base_learning_rate,
                'specialist_learning_rate': self.config.specialist_learning_rate
            },
            'training_state': {
                'base_step': self.base_step,
                'specialist_steps': self.specialist_steps,
                'base_update_count': self.base_update_count,
                'base_accepted_count': self.base_accepted_count,
                'base_rejected_count': self.base_rejected_count,
                'base_baseline_performance': self.base_baseline_performance
            },
            'stats': self.get_system_stats(),
            'timestamp': datetime.now().isoformat()
        }

        serializable_state = _to_serializable(swarm_state)

        with open(checkpoint_dir / 'swarm_state.json', 'w') as f:
            json.dump(serializable_state, f, indent=2)

        print(f"[AdaptiveSwarmTRM] Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: Path):
        """Load complete swarm checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)

        # Load base system
        self.base.load_all(checkpoint_dir)

        # Load swarm state
        with open(checkpoint_dir / 'swarm_state.json', 'r') as f:
            swarm_state = json.load(f)

        # Restore training state
        training_state = swarm_state.get('training_state', {})
        self.base_step = training_state.get('base_step', 0)
        self.specialist_steps = training_state.get('specialist_steps', {})
        self.base_update_count = training_state.get('base_update_count', 0)
        self.base_accepted_count = training_state.get('base_accepted_count', 0)
        self.base_rejected_count = training_state.get('base_rejected_count', 0)
        self.base_baseline_performance = training_state.get('base_baseline_performance', 0.0)

        # Reinitialize shadow weights
        self.W_base_shadow = np.zeros_like(self.base.W_base_full)

        print(f"[AdaptiveSwarmTRM] Checkpoint loaded from {checkpoint_dir}")

    def _compute_base_gradient(self, sample: Dict) -> np.ndarray:
        """
        Compute gradient for base model.

        Placeholder - real implementation would compute actual gradient.

        Args:
            sample: Training sample

        Returns:
            Gradient [D×D]
        """
        # Placeholder: Random gradient (replace with actual backprop)
        return np.random.randn(self.base.max_dims, self.base.max_dims).astype(np.float32) * 0.01

    def _compute_specialist_gradient(self, specialist_name: str, sample: Dict) -> np.ndarray:
        """
        Compute gradient for specialist adapter.

        Args:
            specialist_name: Specialist name
            sample: Training sample

        Returns:
            Gradient [D×D]
        """
        dims = self.base.specialists[specialist_name]['dims']

        # Placeholder: Random gradient (replace with actual backprop)
        return np.random.randn(dims, dims).astype(np.float32) * 0.01

    def _validate_and_commit_base(self, eval_fn: Callable[[np.ndarray, List], float]) -> bool:
        """
        Validate shadow base weights and commit if improved.

        Args:
            eval_fn: Evaluation function

        Returns:
            True if accepted, False if rejected
        """
        # Evaluate baseline
        baseline_perf = eval_fn(self.base.W_base_full, self.base_validation_samples)

        # Evaluate shadow
        shadow_perf = eval_fn(self.W_base_shadow, self.base_validation_samples)

        improvement = shadow_perf - baseline_perf

        self.base_update_count += 1

        # Check for improvement
        min_improvement = 0.001  # 0.1%

        if improvement >= min_improvement:
            # Accept update
            np.copyto(self.base.W_base_full, self.W_base_shadow)
            self.base_baseline_performance = shadow_perf
            self.base_accepted_count += 1

            print(f"[Base] ✓ Update accepted: {baseline_perf:.4f} → {shadow_perf:.4f} (+{improvement:.4f})")
            return True
        else:
            # Reject update
            self.base_rejected_count += 1

            print(f"[Base] ✗ Update rejected: {baseline_perf:.4f} → {shadow_perf:.4f} (+{improvement:.4f})")
            return False


class SwarmTrainingProtocol:
    """
    Training protocol for adaptive swarm.

    Implements standard training workflows:
    1. Base-first: Train base, then specialists
    2. Parallel: Train base and specialists simultaneously
    3. Sequential: Base → Specialist1 → Specialist2 → ...
    """

    @staticmethod
    def train_base_first(swarm: AdaptiveSwarmTRM,
                        general_samples: List[Dict],
                        specialist_samples: Dict[str, List[Dict]],
                        eval_fn: Callable[[np.ndarray, List], float]) -> Dict[str, Any]:
        """
        Train base first, then specialists.

        Args:
            swarm: Adaptive swarm instance
            general_samples: General reasoning samples for base
            specialist_samples: Dict mapping specialist_name → samples
            eval_fn: Evaluation function

        Returns:
            Training statistics
        """
        print("\n" + "="*80)
        print("Training Protocol: Base-First")
        print("="*80)

        # Split general samples
        split_idx = int(len(general_samples) * (1 - swarm.config.validation_split))
        train_general = general_samples[:split_idx]
        val_general = general_samples[split_idx:]

        swarm.set_base_validation_samples(val_general)

        # Train base
        print("\n[Phase 1] Training base model on general reasoning...")
        base_stats = swarm.train_base_epoch(train_general, eval_fn)

        # Train each specialist
        specialist_stats = {}

        for specialist_name, samples in specialist_samples.items():
            print(f"\n[Phase 2] Training specialist '{specialist_name}'...")

            # Split samples
            split_idx = int(len(samples) * (1 - swarm.config.validation_split))
            train_spec = samples[:split_idx]
            val_spec = samples[split_idx:]

            swarm.set_specialist_validation_samples(specialist_name, val_spec)

            # Train
            spec_stats = swarm.train_specialist_epoch(specialist_name, train_spec, eval_fn)
            specialist_stats[specialist_name] = spec_stats

        return {
            'base': base_stats,
            'specialists': specialist_stats
        }
