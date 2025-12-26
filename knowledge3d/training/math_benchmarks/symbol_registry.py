"""
Symbol Registry - Central reference for all symbols across galaxies.

Implements the symlink pattern from FOUNDATIONAL_KNOWLEDGE_SPECIFICATION:
- Each symbol stored ONCE at Layer 1
- Other layers reference by ID (Unicode/codepoint)
- Enables cross-domain discovery via shared references
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class SymbolReference:
    """A reference to a canonical symbol."""

    symbol_id: int  # Unicode codepoint or hash
    symbol: str  # The symbol string (e.g., "∑")
    domains: List[str] = field(default_factory=list)
    rule_refs: List[str] = field(default_factory=list)

    @property
    def cross_domain_score(self) -> float:
        """Higher score = more cross-domain usage."""
        return len(self.domains) * len(self.rule_refs or [1])


class SymbolRegistry:
    """
    Central registry for symbol references.

    Enables:
    1. Symlink pattern: rules reference symbols by ID
    2. Cross-domain discovery: find symbols used across multiple domains
    3. Compression: large savings versus duplicated glyph data
    """

    def __init__(self) -> None:
        self._symbols: Dict[int, SymbolReference] = {}
        self._by_domain: Dict[str, Set[int]] = defaultdict(set)

    def register_symbol(self, symbol_id: int, symbol: str, domain: str, rule_id: str = "") -> None:
        """Register a symbol usage."""
        if symbol_id not in self._symbols:
            self._symbols[symbol_id] = SymbolReference(
                symbol_id=symbol_id, symbol=symbol, domains=[domain], rule_refs=[rule_id] if rule_id else []
            )
        else:
            ref = self._symbols[symbol_id]
            if domain not in ref.domains:
                ref.domains.append(domain)
            if rule_id and rule_id not in ref.rule_refs:
                ref.rule_refs.append(rule_id)

        self._by_domain[domain].add(symbol_id)

    def get_cross_domain_symbols(self, min_domains: int = 2) -> List[SymbolReference]:
        """Find symbols used in multiple domains."""
        return sorted(
            [s for s in self._symbols.values() if len(s.domains) >= min_domains],
            key=lambda s: s.cross_domain_score,
            reverse=True,
        )

    def get_symbols_for_domain(self, domain: str) -> List[SymbolReference]:
        """Get all symbols used in a domain."""
        return [self._symbols[sid] for sid in self._by_domain.get(domain, [])]

    def compression_stats(self) -> Dict[str, object]:
        """Calculate compression statistics (rough symlink savings)."""
        total_refs = sum(len(s.rule_refs) for s in self._symbols.values())
        unique_symbols = len(self._symbols)

        without_symlinks = total_refs * 5120  # Assume 5KB per visual blob
        with_symlinks = unique_symbols * 5120 + total_refs * 4

        return {
            "unique_symbols": unique_symbols,
            "total_references": total_refs,
            "without_symlinks_bytes": without_symlinks,
            "with_symlinks_bytes": with_symlinks,
            "compression_ratio": without_symlinks / max(1, with_symlinks),
            "cross_domain_symbols": len(self.get_cross_domain_symbols(2)),
        }


# Global instance
SYMBOL_REGISTRY = SymbolRegistry()


def populate_registry_from_galaxy() -> None:
    """Populate registry from Math Symbol Galaxy."""
    try:
        from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    except Exception:
        return
    try:
        from knowledge3d.training.arc_agi.math_grammar_rules import get_all_math_rules
    except Exception:
        get_all_math_rules = None  # type: ignore

    for symbol in MATH_GALAXY.all_symbols():
        symbol_id = ord(symbol.symbol[0]) if len(symbol.symbol) == 1 else hash(symbol.symbol)
        SYMBOL_REGISTRY.register_symbol(
            symbol_id=symbol_id,
            symbol=symbol.symbol,
            domain=symbol.category,
            rule_id=f"math_{symbol.symbol}",
        )

    if get_all_math_rules:
        for rule in get_all_math_rules():
            for sym_id in rule.symbol_refs:
                SYMBOL_REGISTRY.register_symbol(
                    symbol_id=sym_id,
                    symbol=chr(sym_id) if sym_id < 0x110000 else str(sym_id),
                    domain=rule.domain,
                    rule_id=rule.rule_id,
                )
