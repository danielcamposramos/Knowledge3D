from __future__ import annotations

"""
Simple charting helper for confidence propagation diagnostics.
"""

from typing import Iterable

try:
    import matplotlib.pyplot as plt  # type: ignore

    _HAS_MPL = True
except Exception:  # pragma: no cover
    plt = None  # type: ignore
    _HAS_MPL = False


def plot_confidence(curiosity: Iterable[float], biased: Iterable[float]) -> None:
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required for visualization helpers")
    curiosity_list = list(curiosity)
    biased_list = list(biased)
    if len(curiosity_list) != len(biased_list):
        raise ValueError("curiosity and biased sequences must have same length")
    if not curiosity_list:
        raise ValueError("no data supplied")
    indices = range(len(curiosity_list))
    plt.figure(figsize=(6, 3))
    plt.plot(indices, curiosity_list, label="Curiosity", marker="o")
    plt.plot(indices, biased_list, label="Biased confidence", marker="x")
    plt.legend()
    plt.title("Confidence Propagation")
    plt.xlabel("Action index")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
