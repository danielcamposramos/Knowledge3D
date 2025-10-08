from __future__ import annotations

"""
Bar chart helper for multi-modal α metrics.
"""

from typing import Dict

try:
    import matplotlib.pyplot as plt  # type: ignore

    _HAS_MPL = True
except Exception:  # pragma: no cover
    plt = None  # type: ignore
    _HAS_MPL = False


def plot_recommendations(recommendations: Dict[str, Dict[str, float]]) -> None:
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required for visualization helpers")
    if not recommendations:
        raise ValueError("no recommendations supplied")
    actions = list(recommendations.keys())
    alphas = [recommendations[a].get("mean_alpha", 0.0) for a in actions]
    plt.figure(figsize=(6, 3))
    bars = plt.bar(actions, alphas)
    for bar, alpha in zip(bars, alphas):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{alpha:.3f}", ha="center", va="bottom")
    plt.ylabel("α value")
    plt.title("Mean α per action")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
