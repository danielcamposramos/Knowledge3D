from __future__ import annotations

"""
Lightweight reinforcement-style optimiser for curiosity bias.

The full Step7.2 blueprint calls for a GPU RL optimiser.  For the current
prototype we implement a deterministic EMA-based optimiser that can be used
from both CPU unit tests and GPU pipelines.  Rewards close to +1 push the
adaptive α upwards, while negative rewards decrease it.  Values are clamped to
the provided range.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class AlphaRange:
    base: float
    minimum: float
    maximum: float


@dataclass
class AlphaState:
    value: float
    ema_reward: float = 0.0
    decay: float = 0.9

    def update(self, reward: float, min_val: float, max_val: float) -> float:
        self.ema_reward = self.decay * self.ema_reward + (1.0 - self.decay) * reward
        self.value = min(max(self.value + self.ema_reward * 0.05, min_val), max_val)
        return self.value


class AlphaRLOptimizer:
    """
    Deterministic optimiser managing α per action type.
    """

    def __init__(self, action_ranges: Dict[str, AlphaRange]) -> None:
        self._ranges = action_ranges
        self._state: Dict[str, AlphaState] = {
            action: AlphaState(value=r.base) for action, r in action_ranges.items()
        }

    def get_alpha(self, action: str) -> float:
        return self._state[action].value

    def update(self, action: str, reward: float) -> float:
        if action not in self._state:
            raise KeyError(f"Unknown action '{action}'")
        r = self._ranges[action]
        return self._state[action].update(reward, r.minimum, r.maximum)
