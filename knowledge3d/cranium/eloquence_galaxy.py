"""
Eloquence Galaxy — Layer 4 meta-rules (strategy, pedagogy, self-reflection).

References Layer 3 grammar rules via rule_refs (symlink pattern).
No duplication of rules or symbols.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional
import json


@dataclass
class MetaRule:
    """Meta-rule referencing Layer 3 grammar rules."""

    meta_id: str
    category: str  # eloquence, pedagogy, self_reflection, storytelling, delivery
    condition: str  # RPN predicate (when to apply)
    action: str     # RPN program (what to do)
    rule_refs: List[str] = field(default_factory=list)  # symlinks to grammar rules
    priority: float = 1.0
    description: str = ""

    def validate_rule_refs(self) -> bool:
        """Ensure referenced rules exist in Grammar Galaxy."""
        try:
            from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
            grammar = GrammarGalaxy()
            for ref in self.rule_refs:
                getter = getattr(grammar, "get_rule", None)
                rule = getter(ref) if callable(getter) else None
                if rule is None and hasattr(grammar, "rules"):
                    rule = grammar.rules.get(ref)  # type: ignore[attr-defined]
                if rule is None:
                    return False
            return True
        except Exception:
            return False


class EloquenceGalaxy:
    """Layer 4 storage for meta-rules."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("/K3D/Knowledge3D.local/galaxies/eloquence")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._meta_rules: Dict[str, MetaRule] = {}
        self._load()

    def _meta_file(self) -> Path:
        return self.storage_path / "meta_rules.json"

    def _load(self) -> None:
        meta_file = self._meta_file()
        if meta_file.exists():
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            for item in data:
                meta = MetaRule(**item)
                self._meta_rules[meta.meta_id] = meta

    def _save(self) -> None:
        meta_file = self._meta_file()
        data = [asdict(m) for m in self._meta_rules.values()]
        meta_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_meta_rule(self, meta: MetaRule) -> bool:
        if not meta.validate_rule_refs():
            raise ValueError(f"Invalid rule_refs for meta-rule {meta.meta_id}")
        self._meta_rules[meta.meta_id] = meta
        self._save()
        return True

    def get(self, meta_id: str) -> Optional[MetaRule]:
        return self._meta_rules.get(meta_id)

    def get_by_category(self, category: str) -> List[MetaRule]:
        return [m for m in self._meta_rules.values() if m.category == category]

    def stats(self) -> Dict[str, object]:
        total = len(self._meta_rules)
        categories = sorted(set(m.category for m in self._meta_rules.values()))
        avg_refs = (
            sum(len(m.rule_refs) for m in self._meta_rules.values()) / total
            if total
            else 0.0
        )
        return {
            "total_meta_rules": total,
            "categories": categories,
            "avg_rule_refs": avg_refs,
        }


# Singleton accessor
_eloquence_galaxy: Optional[EloquenceGalaxy] = None


def get_eloquence_galaxy() -> EloquenceGalaxy:
    global _eloquence_galaxy
    if _eloquence_galaxy is None:
        _eloquence_galaxy = EloquenceGalaxy()
    return _eloquence_galaxy
