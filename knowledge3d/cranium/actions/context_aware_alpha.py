from __future__ import annotations

"""
Context-aware α heuristics.

The advanced design in Step7.2 mentions adapting α based on recent context
embeddings.  The present module implements a light-weight heuristic: store the
most recent context norm per action type and nudge α upwards when we encounter
surprising (high-norm) contexts.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable
import numpy as np


@dataclass
class ContextHistory:
    maxlen: int = 10
    history: Deque[float] = field(default_factory=lambda: deque(maxlen=10))

    def push(self, embedding: Iterable[float]) -> float:
        norm = float(np.linalg.norm(list(embedding)))
        self.history.append(norm)
        return norm

    def std(self) -> float:
        if not self.history:
            return 0.0
        arr = np.asarray(self.history, dtype=np.float32)
        return float(arr.std())


class ContextAwareAlpha:
    """
    Maintains a tiny context history per action type.
    """

    def __init__(self, max_history: int = 10) -> None:
        self._histories: Dict[str, ContextHistory] = {}
        self._max = max_history

    def record(self, action: str, embedding: Iterable[float]) -> float:
        history = self._histories.setdefault(
            action, ContextHistory(maxlen=self._max)
        )
        return history.push(embedding)

    def anomaly_score(self, action: str) -> float:
        history = self._histories.get(action)
        if history is None:
            return 0.0
        return history.std()
