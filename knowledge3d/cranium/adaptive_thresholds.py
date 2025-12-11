"""
Adaptive thresholds backed by Galaxy parameters (no hardcoded constants).
"""

from __future__ import annotations

from typing import Any

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy


class AdaptiveThresholds:
    """
    Lightweight wrapper that stores learned parameters inside GrammarGalaxy.

    NOTE: GrammarGalaxy does not yet persist parameters; we store them in a
    volatile dict on the instance so the interface is ready for PTX-backed
    persistence.
    """

    def __init__(self, galaxy: GrammarGalaxy) -> None:
        self.galaxy = galaxy
        if not hasattr(self.galaxy, "_parameters"):
            setattr(self.galaxy, "_parameters", {})

    def get(self, name: str, default: Any) -> Any:
        return getattr(self.galaxy, "_parameters", {}).get(name, default)

    def set(self, name: str, value: Any) -> None:
        params = getattr(self.galaxy, "_parameters", {})
        params[name] = value
        setattr(self.galaxy, "_parameters", params)

    # Convenience accessors for common knobs
    @property
    def reward_threshold(self) -> float:
        return float(self.get("reward_threshold", 0.85))

    @property
    def neutral_threshold(self) -> float:
        return float(self.get("neutral_threshold", 0.70))

    @property
    def candidate_top_k(self) -> int:
        return int(self.get("candidate_top_k", 69))


__all__ = ["AdaptiveThresholds"]
