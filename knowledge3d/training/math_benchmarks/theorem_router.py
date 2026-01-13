"""
Theorem Router (bootstrap + learned hook).

Phase 1: heuristic routing based on semantic tag overlap.
Phase 2: load learned weights (router specialist outputs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional
import json


class TheoremRouter:
    def __init__(
        self,
        rule_ids: Iterable[str],
        *,
        strategy: str = "heuristic",
        learned_weights_path: Optional[str] = None,
    ) -> None:
        self.rule_ids = [str(r) for r in rule_ids if str(r)]
        self.strategy = str(strategy or "heuristic").lower()
        self.learned_weights_path = str(learned_weights_path) if learned_weights_path else None
        self._learned = self._load_learned_weights()

    def _load_learned_weights(self) -> Dict[str, Dict[str, float]]:
        if not self.learned_weights_path:
            return {}
        path = Path(self.learned_weights_path)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): {str(r): float(v) for r, v in (weights or {}).items()} for k, weights in data.items()}

    @staticmethod
    def _tokenize_tags(tags: Iterable[str]) -> List[str]:
        tokens: List[str] = []
        for tag in tags:
            tok = str(tag or "").strip().lower()
            if tok:
                tokens.append(tok)
        return tokens

    @staticmethod
    def _tokenize_rule(rule_id: str) -> List[str]:
        clean = str(rule_id or "").strip().lower()
        if clean.startswith("theorem:"):
            clean = clean.split(":", 1)[1]
        parts = [p for p in clean.replace("-", "_").split("_") if p]
        return parts

    def route(self, semantic_tags: Iterable[str]) -> Dict[str, float]:
        tags = self._tokenize_tags(semantic_tags)
        if not tags or not self.rule_ids:
            return {}

        if self.strategy == "learned" and self._learned:
            key = "|".join(sorted(tags))
            learned = self._learned.get(key)
            if learned:
                return learned

        # Heuristic bootstrap: overlap between tags and rule_id tokens.
        tag_set = set(tags)
        scores: Dict[str, float] = {}
        for rule_id in self.rule_ids:
            tokens = set(self._tokenize_rule(rule_id))
            overlap = len(tag_set.intersection(tokens))
            scores[rule_id] = float(overlap) if overlap else 0.05
        total = sum(scores.values())
        if total <= 0:
            return {}
        return {rid: score / total for rid, score in scores.items()}

    def select_top(self, semantic_tags: Iterable[str]) -> Optional[str]:
        weights = self.route(semantic_tags)
        if not weights:
            return None
        return max(weights.items(), key=lambda kv: kv[1])[0]


__all__ = ["TheoremRouter"]
