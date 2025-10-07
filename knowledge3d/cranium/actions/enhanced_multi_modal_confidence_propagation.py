from __future__ import annotations

"""
Enhanced multi-modal confidence propagation.

This builds on ``MultiModalConfidencePropagator`` adding the advanced optimiser
and convergence analyser described in Step7.2.  The implementation remains
lightweight while exposing the same API so other modules can consume richer
metrics when they become available.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple
import numpy as np

from .action_types import ActionType
from .multi_modal_confidence_propagation import (
    DEFAULT_ALPHA_RANGES,
    MultiModalConfidencePropagator,
)
from .advanced_alpha_rl_optimizer import AdvancedAlphaRLOptimizer
from .adaptive_convergence_analyzer import AdaptiveConvergenceAnalyzer


@dataclass
class EnhancedMetrics:
    alpha: float
    stability: Dict[str, float] = field(default_factory=dict)


class EnhancedMultiModalConfidencePropagator(MultiModalConfidencePropagator):
    """
    Extends the base propagator with convergence tracking.
    """

    def __init__(self) -> None:
        super().__init__()
        self._alpha_optimizer = AdvancedAlphaRLOptimizer(DEFAULT_ALPHA_RANGES)
        self._analyzer = AdaptiveConvergenceAnalyzer(window=100)

    def propagate_single(
        self,
        action_type: ActionType,
        base_confidence: float,
        curiosity_score: float,
        input_confidence: float,
        context_embedding: Iterable[float],
    ) -> Tuple[float, EnhancedMetrics]:
        biased, alpha = super().propagate_single(
            action_type, base_confidence, curiosity_score, input_confidence, context_embedding
        )
        self._analyzer.update(alpha)
        metrics = EnhancedMetrics(alpha=alpha, stability=self._analyzer.metrics())
        return biased, metrics

    def update_rl(self, action_type: ActionType, reward: float, context_score: float = 0.0) -> float:
        action_name = action_type.name
        return self._alpha_optimizer.update_with_context(action_name, reward, context_score)
