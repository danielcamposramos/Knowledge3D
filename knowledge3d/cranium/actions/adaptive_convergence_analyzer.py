from __future__ import annotations

"""
Adaptive convergence analyser for curiosity-biased confidence.

The Step7.2 blueprint calls for tooling that tracks how quickly the adaptive
α parameters converge.  This module implements a small sliding-window analyser
that can be used both in unit tests and in the demo tooling.  It does not rely
on GPU code, keeping it easy to exercise in CPU-only environments.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import numpy as np

try:  # Optional visualisation dependency
    import matplotlib.pyplot as plt  # type: ignore

    _HAS_MPL = True
except Exception:  # pragma: no cover - fallback when matplotlib unavailable
    plt = None  # type: ignore
    _HAS_MPL = False


@dataclass
class AdaptiveConvergenceAnalyzer:
    """Tracks α values over time and reports stability metrics."""

    window: int = 50
    history: List[float] = field(default_factory=list)

    def update(self, alpha: float) -> None:
        """Record a new α value."""
        self.history.append(float(alpha))
        if len(self.history) > self.window:
            del self.history[0 : len(self.history) - self.window]

    def extend(self, alphas: Iterable[float]) -> None:
        for value in alphas:
            self.update(float(value))

    def metrics(self) -> Dict[str, float]:
        """Return variance / range metrics for the current window."""
        if not self.history:
            return {"variance": 0.0, "range": 0.0, "mean": 0.0}
        arr = np.asarray(self.history, dtype=np.float32)
        return {
            "variance": float(arr.var()),
            "range": float(arr.max() - arr.min()),
            "mean": float(arr.mean()),
        }

    def is_stable(self, threshold: float = 0.01) -> bool:
        """Simple stability check based on variance."""
        return self.metrics()["variance"] <= threshold

    def visualize(self) -> None:
        """Plot the current α history if matplotlib is available."""
        if not _HAS_MPL:
            raise RuntimeError("matplotlib is required for visualize()")
        if not self.history:
            raise RuntimeError("no history recorded yet")

        plt.figure(figsize=(6, 3))
        plt.plot(self.history, marker="o")
        plt.title("Adaptive α history")
        plt.xlabel("Iteration")
        plt.ylabel("α value")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
