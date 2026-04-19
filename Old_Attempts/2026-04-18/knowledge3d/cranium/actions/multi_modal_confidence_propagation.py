from __future__ import annotations

"""
Multi-modal confidence propagation manager.

Combines curiosity bias, RL-controlled α and context heuristics to provide a
single hook the output router can use to adjust confidences.  This is a slimmed
down version of the rich design in Step7.2 but keeps the same surface API so we
can extend it incrementally.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
import numpy as np

from .action_types import ActionType
from .alpha_rl_optimizer import AlphaRLOptimizer, AlphaRange
from .confidence_propagation import ConfidencePropagator
from .context_aware_alpha import ContextAwareAlpha


DEFAULT_ALPHA_RANGES: Dict[str, AlphaRange] = {
    "NAV_MOVE": AlphaRange(base=0.10, minimum=0.05, maximum=0.25),
    "DIALOGUE": AlphaRange(base=0.08, minimum=0.04, maximum=0.20),
    "WRITE_MEM": AlphaRange(base=0.06, minimum=0.02, maximum=0.15),
    "UPDATE_TABLET": AlphaRange(base=0.07, minimum=0.03, maximum=0.18),
}


class MultiModalConfidencePropagator:
    """
    Entry point used by the action router.
    """

    def __init__(self) -> None:
        self._alpha_optimizer = AlphaRLOptimizer(DEFAULT_ALPHA_RANGES)
        self._context = ContextAwareAlpha(max_history=10)
        self._base_prop = ConfidencePropagator()

    def calculate_adaptive_alpha(
        self,
        action_type: ActionType,
        base_confidence: float,
        context_embedding: Iterable[float],
    ) -> float:
        action_name = action_type.name
        norm = self._context.record(action_name, context_embedding)
        anomaly = self._context.anomaly_score(action_name)

        alpha = self._alpha_optimizer.get_alpha(action_name)
        if anomaly > 1.0:
            alpha = min(alpha + anomaly * 0.01, DEFAULT_ALPHA_RANGES[action_name].maximum)
        return alpha

    def propagate_single(
        self,
        action_type: ActionType,
        base_confidence: float,
        curiosity_score: float,
        input_confidence: float,
        context_embedding: Iterable[float],
    ) -> Tuple[float, float]:
        alpha = self.calculate_adaptive_alpha(action_type, base_confidence, context_embedding)
        biased = self._base_prop.propagate_confidence(
            [base_confidence],
            [curiosity_score * alpha],
            input_confidence,
        )[0]
        return biased, alpha

    def update_rl(self, action_type: ActionType, reward: float) -> float:
        action_name = action_type.name
        return self._alpha_optimizer.update(action_name, reward)
