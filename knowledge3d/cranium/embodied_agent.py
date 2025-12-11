"""
EmbodiedSovereignAgent — persists Galaxy state and maintains working memory.
"""

from __future__ import annotations

from typing import List, Sequence, Dict, Optional

from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
from knowledge3d.cranium.word_galaxy import get_word_galaxy, WordGalaxy
from knowledge3d.cranium.eloquence_galaxy import get_eloquence_galaxy, EloquenceGalaxy
from knowledge3d.cranium.math_galaxy import get_math_galaxy
from knowledge3d.cranium.ternary_working_memory import TernaryWorkingMemory
from knowledge3d.cranium.adaptive_thresholds import AdaptiveThresholds
from knowledge3d.cranium.semantic_rule_bridge import SemanticRuleBridge


class EmbodiedSovereignAgent:
    """
    Persistent agent that keeps galaxies loaded once and accrues discoveries
    in a ternary working memory buffer.
    """

    def __init__(self, *, working_capacity: int = 4096) -> None:
        self.drawing_galaxy = DrawingGalaxy()
        self.grammar_galaxy = GrammarGalaxy()
        self.word_galaxy: WordGalaxy = get_word_galaxy()
        self.eloquence_galaxy: EloquenceGalaxy = get_eloquence_galaxy()
        self.math_galaxy = get_math_galaxy()
        self.working_memory = TernaryWorkingMemory(capacity=working_capacity)
        self.thresholds = AdaptiveThresholds(self.grammar_galaxy)
        self.semantic_bridge = SemanticRuleBridge(
            rule_ids=list(self.grammar_galaxy.rules.keys()),
            top_k=self.thresholds.candidate_top_k,
        )

    def should_consolidate(self) -> bool:
        return self.working_memory.utilization > 0.85

    def consolidate(self) -> None:
        """Deduplicate and clear working memory (promotion hook TBD)."""
        _ = self.working_memory.deduplicate()
        self.working_memory.clear()

    def work_on_task(self, task_id: str, grid: Sequence[Sequence[int]]) -> Dict:
        """
        Skeleton for embodied task processing. Currently records the grid hash
        into working memory; execution path stays unchanged.
        """
        signature = {"task_id": task_id, "grid_h": len(grid), "grid_w": len(grid[0]) if grid else 0}
        self.working_memory.add(signature)
        return signature


__all__ = ["EmbodiedSovereignAgent"]
