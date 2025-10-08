from __future__ import annotations

"""
Quick Matplotlib helper to visualise adaptive convergence metrics.
"""

from typing import Iterable

try:
    import matplotlib.pyplot as plt  # type: ignore

    _HAS_MPL = True
except Exception:  # pragma: no cover
    plt = None  # type: ignore
    _HAS_MPL = False


def plot_history(history: Iterable[float]) -> None:
    """Plot α history in a compact 2 × 1 layout."""
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required for visualization helpers")
    data = list(history)
    if not data:
        raise ValueError("no history provided")
    plt.figure(figsize=(6, 3))
    plt.plot(data, marker="o")
    plt.title("Adaptive α History")
    plt.xlabel("Iteration")
    plt.ylabel("α value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
