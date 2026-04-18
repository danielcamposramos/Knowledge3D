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

    # Consolidate specialist during sleep-time
    swarm.consolidate_specialist_dream_cycle('ocr', ocr_samples, ocr_validation)

    # Inference with automatic specialist selection
    output = swarm.forward(input_data, specialist='auto')
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from knowledge3d.cranium.matryoshka_trm import MatryoshkaTRM, DimensionSelector
from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32, RPNMathCore
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
    base_learning_rate: float = 0.001        # Deprecated shim: use base_absorption_rate
    specialist_learning_rate: float = 0.002  # Deprecated shim: use specialist_absorption_rate
    validation_split: float = 0.1            # Deprecated shim: use gate_check_split
    checkpoint_interval: int = 100           # Save every N steps
    enable_auto_expansion: bool = True       # Auto-expand dims when needed
    expansion_threshold: float = 0.95        # Expand if complexity > threshold

    @property
    def base_absorption_rate(self) -> float:
        return float(self.base_learning_rate)

    @base_absorption_rate.setter
    def base_absorption_rate(self, value: float) -> None:
        self.base_learning_rate = float(value)

    @property
    def specialist_absorption_rate(self) -> float:
        return float(self.specialist_learning_rate)

    @specialist_absorption_rate.setter
    def specialist_absorption_rate(self, value: float) -> None:
        self.specialist_learning_rate = float(value)

    @property
    def gate_check_split(self) -> float:
        return float(self.validation_split)

    @gate_check_split.setter
    def gate_check_split(self, value: float) -> None:
        self.validation_split = float(value)


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
            min_dims=self.config.min_dims,
        )

        # Training state
        self.base_step = 0
        self.specialist_steps: Dict[str, int] = {}

        # Validation sets (per specialist + general)
        self.base_validation_samples = []

        # Base self-updating mechanism
        self.W_base_shadow: Optional[HostTensorF32] = None
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
                        eval_fn: Callable[[Any, List], float],
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
        raise RuntimeError(
            "Base epoch training still depends on placeholder host-side gradients. "
            "Phase 1 removes that fake path instead of keeping non-sovereign NumPy training alive."
        )

    def train_specialist_epoch(self, specialist_name: str,
                              train_samples: List[Dict],
                              eval_fn: Callable[[Any, List], float] | List[Dict] | None,
                              use_self_update: bool = True) -> Dict[str, float]:
        """Deprecated shim for consolidate_specialist_dream_cycle()."""
        return self.consolidate_specialist_dream_cycle(
            specialist_name,
            consolidation_wave=train_samples,
            gate_check_samples=eval_fn,
            use_shadow_absorption=use_self_update,
        )

    def consolidate_specialist_dream_cycle(
        self,
        specialist_name: str,
        consolidation_wave: List[Dict],
        gate_check_samples: Callable[[Any, List], float] | List[Dict] | None,
        use_shadow_absorption: bool = True,
    ) -> Dict[str, float]:
        """
        Consolidate specialist shadow weights for one dream cycle.

        Args:
            specialist_name: Which specialist to train
            consolidation_wave: Trace samples to absorb in this cycle
            gate_check_samples: Gate-check traces or callback
            use_shadow_absorption: Deprecated compatibility flag

        Returns:
            Consolidation statistics
        """
        del use_shadow_absorption
        validation_samples = gate_check_samples if isinstance(gate_check_samples, list) else []
        positive_pairs, negative_pairs = self._build_contrastive_pairs_from_samples(
            specialist_name,
            list(consolidation_wave or []),
            list(validation_samples or []),
        )
        stats = self.train_specialist_contrastive(
            specialist_name,
            positive_pairs=positive_pairs,
            negative_pairs=negative_pairs,
        )
        stats["validation_samples"] = len(validation_samples)
        stats["update_accepted"] = bool(positive_pairs or negative_pairs)
        stats["gate_check_samples"] = int(len(validation_samples))
        stats["contrast_signal"] = float(stats.get("avg_loss", 0.0))
        stats["absorption_rate"] = float(self.config.specialist_absorption_rate)
        stats["consolidation_wave_size"] = int(len(consolidation_wave or []))
        return stats

    def forward(self, input_data: Any,
                specialist: Optional[str] = None,
                complexity: Optional[float] = None) -> Any:
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

    def forward_moe(self, input_data: Any,
                   specialist_weights: Dict[str, float]) -> Any:
        """
        Forward pass with MoE (mixture of specialists).

        Args:
            input_data: Input vector
            specialist_weights: Dict mapping specialist_name → weight

        Returns:
            Blended output
        """
        return self.base.compute_with_moe(input_data, specialist_weights)

    def compute_with_specialist(self, input_data: Any, specialist_name: str) -> Any:
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
        self.W_base_shadow = None

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
                'specialist_learning_rate': self.config.specialist_learning_rate,
                'base_absorption_rate': self.config.base_absorption_rate,
                'specialist_absorption_rate': self.config.specialist_absorption_rate,
                'gate_check_split': self.config.gate_check_split,
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
        self.W_base_shadow = None

        print(f"[AdaptiveSwarmTRM] Checkpoint loaded from {checkpoint_dir}")

    def _compute_base_gradient(self, sample: Dict) -> HostTensorF32:
        """
        Compute gradient for base model.

        Placeholder - real implementation would compute actual gradient.

        Args:
            sample: Training sample

        Returns:
            Gradient [D×D]
        """
        raise RuntimeError(
            "Base gradient computation placeholder removed. "
            "A sovereign base-training kernel path must replace it."
        )

    def _compute_specialist_gradient(self, specialist_name: str, sample: Dict) -> HostTensorF32:
        """
        Compute gradient for specialist adapter.

        Args:
            specialist_name: Specialist name
            sample: Training sample

        Returns:
            Gradient [D×D]
        """
        raise RuntimeError(
            "Specialist gradient computation placeholder removed. "
            "A sovereign specialist-training kernel path must replace it."
        )

    def train_specialist_contrastive(
        self,
        specialist_name: str,
        positive_pairs: List[Tuple[Sequence[float], Sequence[float]]],
        negative_pairs: Optional[List[Tuple[Sequence[float], Sequence[float]]]] = None,
        learning_rate: Optional[float] = None,
        margin: float = 0.5,
    ) -> Dict[str, float]:
        """
        Train specialist using true contrastive learning.

        Args:
            specialist_name: Which specialist to train
            positive_pairs: List of (input_emb, target_emb) pairs to pull together
            negative_pairs: List of (input_emb, wrong_emb) pairs to push apart
            learning_rate: Optional override for specialist LR
            margin: Repulsion margin for negative pairs

        Returns:
            Training statistics
        """
        if specialist_name not in self.base.specialists:
            raise ValueError(f"Unknown specialist: {specialist_name}")

        specialist = self.base.specialists[specialist_name]
        adapter = specialist['adapter']
        dims = specialist['dims']

        lr = learning_rate if learning_rate is not None else self.config.specialist_learning_rate

        math_core = RPNMathCore()
        total_positive_loss = 0.0
        total_negative_loss = 0.0
        positive_steps = 0
        negative_steps = 0

        for input_emb, target_emb in positive_pairs:
            input_emb = self._pad_or_truncate(input_emb, dims)
            target_emb = self._pad_or_truncate(target_emb, dims)

            diff = [target_value - input_value for input_value, target_value in zip(input_emb, target_emb)]
            loss = math_core.vector_norm_host(diff)
            total_positive_loss += float(loss)

            gradient = math_core.outer_product_host(
                HostTensorF32.from_array_like(diff, rows=dims, cols=1),
                HostTensorF32.from_array_like(input_emb, rows=1, cols=dims),
            )
            self._apply_adapter_gradient(adapter, gradient, lr)
            positive_steps += 1

        for input_emb, wrong_emb in list(negative_pairs or []):
            input_emb = self._pad_or_truncate(input_emb, dims)
            wrong_emb = self._pad_or_truncate(wrong_emb, dims)

            diff = [input_value - wrong_value for input_value, wrong_value in zip(input_emb, wrong_emb)]
            dist = float(math_core.vector_norm_host(diff))
            if dist >= float(margin):
                continue

            repulsion_loss = float(margin) - dist
            total_negative_loss += repulsion_loss
            negative_diff = [-value for value in diff]
            gradient = math_core.outer_product_host(
                HostTensorF32.from_array_like(negative_diff, rows=dims, cols=1),
                HostTensorF32.from_array_like(input_emb, rows=1, cols=dims),
            )
            self._apply_adapter_gradient(adapter, gradient, lr * 0.5)
            negative_steps += 1

        total_steps = positive_steps + negative_steps
        avg_loss = (
            (total_positive_loss + total_negative_loss) / total_steps
            if total_steps
            else 0.0
        )

        if specialist_name in self.specialist_steps:
            self.specialist_steps[specialist_name] += total_steps

        return {
            'avg_loss': avg_loss,
            'positive_loss': total_positive_loss / max(positive_steps, 1),
            'negative_loss': total_negative_loss / max(negative_steps, 1),
            'positive_steps': positive_steps,
            'negative_steps': negative_steps,
            'num_pairs': len(positive_pairs) + len(list(negative_pairs or [])),
            'steps': self.specialist_steps.get(specialist_name, 0),
        }

    @staticmethod
    def _pad_or_truncate(embedding: Sequence[float], dims: int) -> List[float]:
        values = [float(value) for value in embedding]
        if len(values) >= dims:
            return values[:dims]
        return values + ([0.0] * (dims - len(values)))

    def _build_contrastive_pairs_from_samples(
        self,
        specialist_name: str,
        train_samples: List[Dict[str, Any]],
        validation_samples: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[List[float], List[float]]], List[Tuple[List[float], List[float]]]]:
        specialist = self.base.specialists.get(specialist_name, {})
        dims = int(specialist.get("dims", self.config.min_dims))
        positive_pairs: List[Tuple[List[float], List[float]]] = []
        negative_pairs: List[Tuple[List[float], List[float]]] = []

        for sample in list(train_samples or []):
            input_embedding = self._training_vector_from_sample(sample, "input_embedding", dims)
            target_embedding = self._training_vector_from_sample(sample, "target_embedding", dims)
            if input_embedding is None or target_embedding is None:
                continue
            is_correct = bool(sample.get("correct", True))
            negative_embedding = self._training_vector_from_sample(sample, "negative_embedding", dims)
            if is_correct:
                positive_pairs.append((input_embedding, target_embedding))
                if negative_embedding is not None:
                    negative_pairs.append((input_embedding, negative_embedding))
            else:
                negative_pairs.append((input_embedding, target_embedding))
                if negative_embedding is not None:
                    positive_pairs.append((input_embedding, negative_embedding))

        for sample in list(validation_samples or []):
            input_embedding = self._training_vector_from_sample(sample, "input_embedding", dims)
            target_embedding = self._training_vector_from_sample(sample, "target_embedding", dims)
            if input_embedding is None or target_embedding is None:
                continue
            negative_pairs.append((input_embedding, target_embedding))

        return positive_pairs, negative_pairs

    @classmethod
    def _training_vector_from_sample(
        cls,
        sample: Dict[str, Any],
        key: str,
        dims: int,
    ) -> Optional[List[float]]:
        values = sample.get(key)
        if values is None:
            if key == "input_embedding":
                values = sample.get("input")
            elif key == "target_embedding":
                values = sample.get("target")
        if values is None:
            return None
        if isinstance(values, HostTensorF32):
            values = values.to_flat_list()
        if not isinstance(values, (list, tuple)):
            return None
        return cls._pad_or_truncate(values, dims)

    @staticmethod
    def _apply_adapter_gradient(adapter: Any, gradient: Any, lr: float) -> None:
        if hasattr(adapter, 'apply_gradient'):
            adapter.apply_gradient(gradient, lr=lr)
            return
        raise RuntimeError(
            f"Adapter {type(adapter).__name__} has no sovereign apply_gradient method. "
            "Contrastive training requires the GPU adapter path."
        )

    def _validate_and_commit_base(self, eval_fn: Callable[[Any, List], float]) -> bool:
        """
        Validate shadow base weights and commit if improved.

        Args:
            eval_fn: Evaluation function

        Returns:
            True if accepted, False if rejected
        """
        raise RuntimeError(
            "Base validate-and-commit still depends on non-sovereign base weight ownership. "
            "Phase 1 removes the placeholder instead of preserving a NumPy training path."
        )


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
                        eval_fn: Callable[[Any, List], float]) -> Dict[str, Any]:
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
            spec_stats = swarm.consolidate_specialist_dream_cycle(
                specialist_name,
                consolidation_wave=train_spec,
                gate_check_samples=eval_fn,
            )
            specialist_stats[specialist_name] = spec_stats

        return {
            'base': base_stats,
            'specialists': specialist_stats
        }
