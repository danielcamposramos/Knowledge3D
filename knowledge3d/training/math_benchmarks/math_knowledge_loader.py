"""
Math Knowledge Loader - ingest pre-extracted JSONs into lightweight grammar hints.

Loads mathematical knowledge from EchoSystems Default Libraries:
- Formulas and identities
- Transformation rules
- RPN-native algorithms
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule
from knowledge3d.training.math_benchmarks.rpn_parser import RPNParser
from knowledge3d.training.math_benchmarks.rpn_validator import is_valid_rpn


class MathKnowledgeLoader:
    """Load and parse pre-extracted math knowledge."""

    KNOWLEDGE_BASE = Path("/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/JSON")

    def __init__(self) -> None:
        self.formulas: List[Dict[str, Any]] = []
        self.rules: List[Dict[str, Any]] = []
        self.rpn_patterns: List[Dict[str, Any]] = []
        self._parser = RPNParser()

    def load_all(self) -> Dict[str, int]:
        """Load all available JSON knowledge."""
        stats = {"formulas": 0, "rules": 0, "rpn_patterns": 0}

        self._load_rpn_knowledge()
        self._load_generic_jsons()

        stats["formulas"] = len(self.formulas)
        stats["rules"] = len(self.rules)
        stats["rpn_patterns"] = len(self.rpn_patterns)
        return stats

    def _load_json_lenient(self, path: Path) -> Any:
        """
        Load JSON with a minimal escape-repair fallback.

        Some of the legacy EchoSystems exports contain backslashes that are not
        valid JSON escapes (often due to LaTeX fragments in plain text). This
        ingestion step is allowed to be flexible; inference remains sovereign.
        """
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            repaired = re.sub(r"\\(?![\\\\/\"bfnrtu])", r"\\\\", text)
            return json.loads(repaired)

    def _load_rpn_knowledge(self) -> None:
        """Load RPN-specific knowledge from the RPN papers."""
        rpn_files = [
            "vertopal.com_ReversePolishNotatonMethod.json",
            "vertopal.com_3.3. Reverse Polish - Intermediate.json",
        ]
        for fname in rpn_files:
            path = self.KNOWLEDGE_BASE / fname
            if path.exists():
                try:
                    data = self._load_json_lenient(path)
                except Exception:
                    continue
                self.rpn_patterns.append({"source": fname, "content": data})

    def _load_generic_jsons(self) -> None:
        if not self.KNOWLEDGE_BASE.exists():
            return
        for path in sorted(self.KNOWLEDGE_BASE.glob("*.json")):
            if "ReversePolish" in path.name or "Reverse Polish" in path.name:
                continue  # already handled
            try:
                data = self._load_json_lenient(path)
            except Exception:
                continue
            self._extract_formulas(path.name, data)

    def _infer_domain(self, lhs: str, rhs: str, source: str) -> str:
        """Heuristic domain detection for loaded formulas."""
        txt = f"{lhs} {rhs} {source}".lower()
        if any(k in txt for k in ["interest", "npv", "present value", "future value", "annuity"]):
            return "math_finance"
        if any(k in txt for k in ["d/dx", "∂", "integral", "∫", "derivative"]):
            return "math_calculus"
        if any(k in txt for k in ["matrix", "det", "eigen", "vector"]):
            return "math_linear_algebra"
        if any(k in txt for k in ["area", "volume", "geometry", "triangle", "circle", "sphere"]):
            return "math_geometry"
        if any(k in txt for k in ["probability", "expectation", "variance", "distribution"]):
            return "math_statistics"
        if any(k in txt for k in ["sequence", "series", "fibonacci", "arithmetic progression", "geometric progression"]):
            return "math_sequences"
        if any(k in txt for k in ["percentage", "%", "rate", "ratio"]):
            return "math_arithmetic"
        return "math_kb"

    def _extract_formulas(self, source: str, data: Any) -> None:
        """
        Extract mathematical formulas and convert to simple lhs/rhs pairs.

        Expects JSON entries containing text under 'content' or 'text'.
        """
        entries: List[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    txt = item.get("content") or item.get("text")
                    if isinstance(txt, str):
                        entries.append(txt)
        elif isinstance(data, dict):
            for value in data.values():
                if isinstance(value, str):
                    entries.append(value)

        # Multiple regex patterns for different formula styles
        patterns = [
            # Standard equations (allow longer text)
            re.compile(r"([A-Za-z0-9_\^()\+\-\*/%√∞π\[\] ]{1,120})\s*=\s*([A-Za-z0-9_\^()\+\-\*/%√∞π\[\] ]{1,120})"),
            # Function definitions
            re.compile(r"([a-zA-Z]\([a-zA-Z, ]+\))\s*=\s*(.{1,120})"),
            # Derivatives
            re.compile(r"d/d([a-z])\s*\(?\s*(.{1,80})\s*\)?\s*=\s*(.{1,80})"),
            # Integrals
            re.compile(r"[∫∮]\s*(.{1,80})\s*d([a-z])\s*=\s*(.{1,80})"),
        ]

        seen = set()
        for entry in entries:
            for pattern in patterns:
                for match in pattern.finditer(entry):
                    groups = match.groups()
                    if len(groups) >= 2:
                        lhs = groups[0].strip()
                        rhs = groups[-1].strip()
                        if not lhs or not rhs:
                            continue
                        # Convert RHS to RPN and validate before storing
                        try:
                            rpn = self._parser.infix_to_rpn(rhs)
                            if not is_valid_rpn(rpn):
                                continue
                        except Exception:
                            continue
                        domain = self._infer_domain(lhs, rhs, source)
                        key = (lhs, rhs, domain)
                        if key in seen:
                            continue
                        seen.add(key)
                        self.formulas.append(
                            {
                                "lhs": lhs,
                                "rhs": rhs,
                                "source": source,
                                "domain": domain,
                                "rpn": rpn,
                            }
                        )

    def to_grammar_rules(self) -> List[GrammarRule]:
        """Convert loaded formulas to GrammarRule objects."""
        rules: List[GrammarRule] = []
        for idx, item in enumerate(self.formulas):
            lhs = item["lhs"]
            rhs = item["rhs"]
            domain = item.get("domain", "math_kb")
            rpn = item.get("rpn", "")
            if not rpn:
                try:
                    rpn = self._parser.infix_to_rpn(rhs)
                except Exception:
                    rpn = rhs
            if not is_valid_rpn(rpn):
                continue
            pattern = re.escape(lhs)
            rules.append(
                GrammarRule(
                    rule_id=f"kb_eq_{idx}",
                    language="math",
                    pattern=pattern,
                    rpn_program=rpn,
                    domain=domain,
                    symbol_refs=[],
                    examples=[],
                )
            )
        return rules

    def populate_math_galaxy(self, *, max_symbols: int = 5000) -> int:
        """
        Ingest parsed formulas into MathSymbolGalaxy as derived entries.

        This is an ingestion step (flexible) that expands the galaxy; inference
        stays sovereign by navigating the already-populated entries.
        """
        try:
            from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY, MathSymbol
        except Exception:
            return 0

        added = 0
        limit = max(0, int(max_symbols))
        for idx, item in enumerate(self.formulas):
            if added >= limit:
                break
            rpn = item.get("rpn", "") or ""
            if not rpn or not is_valid_rpn(rpn):
                continue
            lhs = str(item.get("lhs", "")).strip()
            rhs = str(item.get("rhs", "")).strip()
            domain = str(item.get("domain", "math_kb"))
            source = str(item.get("source", ""))

            symbol_id = f"kb_eq_{idx}"
            if MATH_GALAXY.lookup(symbol_id) is not None:
                continue

            MATH_GALAXY.add_symbol(
                MathSymbol(
                    symbol=symbol_id,
                    category=domain,
                    arity=0,
                    rpn_template=rpn,
                    precedence=0,
                    associativity="none",
                    description=lhs or symbol_id,
                    metadata={"lhs": lhs, "rhs": rhs, "source": source, "domain": domain},
                )
            )
            added += 1
        return added

    def prioritized_rules(self, limit: int = 200) -> List[GrammarRule]:
        """Return a prioritized subset (longest LHS first)."""
        return sorted(self.to_grammar_rules(), key=lambda r: len(r.pattern), reverse=True)[:limit]


__all__ = ["MathKnowledgeLoader"]
