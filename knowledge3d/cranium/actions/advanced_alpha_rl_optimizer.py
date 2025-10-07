from __future__ import annotations

"""
Advanced alpha optimiser – lightweight extension of AlphaRLOptimizer.

The fully fledged RL optimiser described in Step7.2 would live on the GPU and
use stochastic policies.  For now we provide a deterministic wrapper that
tracks momentum and optional context scores, which keeps the public API in
place while remaining easy to exercise in unit tests.
"""

from dataclasses import dataclass
from typing import Dict

from .alpha_rl_optimizer import AlphaRLOptimizer, AlphaRange


@dataclass
class MomentumState:
    value: float = 0.0
    momentum: float = 0.8

    def update(self, gradient: float) -> float:
        self.value = self.momentum * self.value + (1.0 - self.momentum) * gradient
        return self.value


class AdvancedAlphaRLOptimizer(AlphaRLOptimizer):
    """
    Extends AlphaRLOptimizer with momentum and context factors.
    """

    def __init__(self, action_ranges: Dict[str, AlphaRange], momentum: float = 0.8) -> None:
        super().__init__(action_ranges)
        self._momentum = {name: MomentumState(momentum=momentum) for name in action_ranges}

    def update_with_context(self, action: str, reward: float, context_score: float) -> float:
        """
        Update α using both reward and a context surprise score.
        """
        trend = self._momentum[action].update(context_score)
        adjusted_reward = reward + 0.05 * trend
        return super().update(action, adjusted_reward)
