"""
Cross-Domain Discovery Layer.

Analyzes shared symbol references across domains to surface emergent connections.
Symbols are canonical (Math Galaxy); rules symlink to them, enabling automatic
generalization across domains.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set

from knowledge3d.cranium.math_galaxy import get_math_galaxy


@dataclass
class CrossDomainConnection:
    """A symbol bridging multiple domains."""

    symbol_id: int
    char: str
    domains: List[str]
    rule_ids: List[str]
    strength: float  # Higher = more rules per domain
    insight: str = ""


class DiscoveryLayer:
    """
    Discovers cross-domain connections via shared symbol references.

    Workflow:
        - register_rule(rule_id, domain, symbol_refs)
        - discover_connections() to compute bridges
        - report() for human-readable summary
    """

    def __init__(self) -> None:
        self.math_galaxy = get_math_galaxy()
        self.symbol_to_rules: Dict[int, List[str]] = defaultdict(list)
        self.symbol_to_domains: Dict[int, Set[str]] = defaultdict(set)

    def register_rule(self, rule_id: str, domain: str, symbol_refs: List[int]) -> None:
        """Register a grammar rule's symbol references."""
        for codepoint in symbol_refs:
            self.symbol_to_rules[codepoint].append(rule_id)
            self.symbol_to_domains[codepoint].add(domain)

    def discover_connections(self, min_domains: int = 2) -> List[CrossDomainConnection]:
        """Find symbols that bridge multiple domains."""
        connections: List[CrossDomainConnection] = []
        for codepoint, domains in self.symbol_to_domains.items():
            if len(domains) < min_domains:
                continue

            symbol = self.math_galaxy.get(codepoint)
            if symbol is None:
                continue

            rule_ids = self.symbol_to_rules[codepoint]
            strength = len(rule_ids) / max(len(domains), 1)
            insight = self._generate_insight(symbol.char, list(domains), getattr(symbol, "name", ""))

            connections.append(
                CrossDomainConnection(
                    symbol_id=codepoint,
                    char=symbol.char,
                    domains=list(domains),
                    rule_ids=rule_ids,
                    strength=strength,
                    insight=insight,
                )
            )
        return sorted(connections, key=lambda c: c.strength, reverse=True)

    def _generate_insight(self, char: str, domains: List[str], name: str) -> str:
        """Generate human-readable insight for a cross-domain connection."""
        domain_names = [d.replace("math_", "").replace("_", " ") for d in domains]
        insights = {
            "∑": f"Iterative accumulation bridges {', '.join(domain_names)}",
            "∫": f"Continuous accumulation bridges {', '.join(domain_names)}",
            "∂": f"Rate of change bridges {', '.join(domain_names)}",
            "∇": f"Gradient/direction bridges {', '.join(domain_names)}",
            "∀": f"Universal quantification bridges {', '.join(domain_names)}",
            "∃": f"Existential assertion bridges {', '.join(domain_names)}",
        }
        return insights.get(char, f"{name or char} connects {', '.join(domain_names)}")

    def report(self, top_k: int = 20) -> str:
        """Generate a discovery report."""
        connections = self.discover_connections()

        lines = [
            "=" * 60,
            "CROSS-DOMAIN DISCOVERY REPORT",
            "=" * 60,
            f"Total symbols analyzed: {len(self.symbol_to_domains)}",
            f"Cross-domain connections: {len(connections)}",
            "",
        ]

        for conn in connections[:top_k]:
            lines.append(f"  {conn.char} (U+{conn.symbol_id:04X})")
            lines.append(f"    Domains: {', '.join(conn.domains)}")
            lines.append(f"    Rules: {len(conn.rule_ids)}")
            lines.append(f"    Strength: {conn.strength:.2f}")
            lines.append(f"    Insight: {conn.insight}")
            lines.append("")

        return "\n".join(lines)
