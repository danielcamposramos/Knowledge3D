"""
SemanticRuleBridge — maps task hints to Drawing/Grammar rules.

This is a scaffold; embeddings and PTX kernels should replace the placeholder
list operations when available.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


def _ngram_signature(text: str, n: int = 3) -> List[str]:
    """Lowercase n-grams for cheap semantic matching."""
    t = text.lower()
    if len(t) < n:
        return [t]
    return [t[i : i + n] for i in range(len(t) - n + 1)]


class SemanticRuleBridge:
    def __init__(self, rule_ids: Sequence[str] | None = None, *, top_k: int = 12) -> None:
        self.rule_ids = list(rule_ids or [])
        self.top_k = top_k

    def encode_hints(self, hints: List[str]) -> List[str]:
        """Normalize and split hints into n-grams."""
        encoded: List[str] = []
        for h in hints:
            if not isinstance(h, str):
                continue
            h_clean = h.strip().lower()
            if not h_clean:
                continue
            encoded.extend(_ngram_signature(h_clean))
        return encoded

    def get_applicable_rules(self, task_hints: List[str]) -> List[str]:
        """
        Heuristic n-gram overlap scoring between hints and rule ids.
        """
        if not task_hints or not self.rule_ids:
            return []
        encoded = self.encode_hints(task_hints)
        scores: List[Tuple[str, int]] = []
        for rid in self.rule_ids:
            rid_ngrams = _ngram_signature(rid)
            overlap = sum(1 for ng in rid_ngrams if ng in encoded)
            if overlap > 0:
                scores.append((rid, overlap))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [rid for rid, _ in scores[: self.top_k]]


__all__ = ["SemanticRuleBridge"]
