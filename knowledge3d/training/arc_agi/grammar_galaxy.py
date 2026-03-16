"""Defeasible-aware compatibility wrapper for the archived ARC grammar galaxy."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys
from typing import Any
import json

from knowledge3d.cranium.math_galaxy import get_math_galaxy


_LEGACY_PATH = (
    Path(__file__).resolve().parents[3]
    / "Old_Attempts"
    / "curriculum_specific_training"
    / "arc_agi"
    / "grammar_galaxy.py"
)
_LEGACY_SPEC = importlib.util.spec_from_file_location(
    "knowledge3d_legacy_arc_grammar_galaxy",
    _LEGACY_PATH,
)
if _LEGACY_SPEC is None or _LEGACY_SPEC.loader is None:
    raise RuntimeError(f"Unable to load legacy grammar galaxy from {_LEGACY_PATH}")
_LEGACY_MODULE = importlib.util.module_from_spec(_LEGACY_SPEC)
sys.modules[_LEGACY_SPEC.name] = _LEGACY_MODULE
_LEGACY_SPEC.loader.exec_module(_LEGACY_MODULE)

_STRICT_RULE_IDS = {
    "apply_power_rule_natural",
    "apply_power_rule_prime",
    "apply_power_rule_leibniz",
    "apply_sum_rule_natural",
    "apply_sum_rule_prime",
    "apply_sum_rule_leibniz",
    "apply_product_rule_natural",
    "apply_product_rule_prime",
    "apply_product_rule_leibniz",
    "apply_fundamental_theorem_calculus_natural",
    "apply_fundamental_theorem_calculus_latex",
}


def _clamp_rule_strength(value: Any, *, rule_id: str = "") -> int:
    try:
        numeric = int(value)
    except Exception:
        numeric = 1 if str(rule_id).strip() in _STRICT_RULE_IDS else 0
    if numeric > 0:
        return 1
    if numeric < 0:
        return -1
    return 0


@dataclass
class GrammarRule:
    rule_id: str
    language: str
    pattern: str
    rpn_program: str
    domain: str = "text"
    symbol_refs: list[Any] = field(default_factory=list)
    word_refs: list[str] = field(default_factory=list)
    examples: list[Any] = field(default_factory=list)
    description: str | None = None
    semantics: dict[str, Any] = field(default_factory=dict)
    usage_conditions: list[str] = field(default_factory=list)
    is_canonical: bool = False
    rule_strength: int = 0
    superior_to: list[str] = field(default_factory=list)
    trust_weight: float = 1.0

    def validate_symbol_refs(self) -> bool:
        math_galaxy = get_math_galaxy()
        for symbol_ref in self.symbol_refs:
            if not isinstance(symbol_ref, int):
                continue
            if math_galaxy.get(symbol_ref) is None:
                return False
        return True


def _coerce_symbol_refs(values: Any) -> list[Any]:
    out: list[Any] = []
    for value in list(values or []):
        if isinstance(value, int):
            out.append(value)
            continue
        try:
            text = str(value).strip()
        except Exception:
            continue
        if not text:
            continue
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            out.append(int(text))
        else:
            out.append(text)
    return out


def _upgrade_rule(rule: Any) -> GrammarRule:
    semantics = getattr(rule, "semantics", {}) or {}
    return GrammarRule(
        rule_id=str(getattr(rule, "rule_id", "")),
        language=str(getattr(rule, "language", "en")),
        pattern=str(getattr(rule, "pattern", "unknown")),
        rpn_program=str(getattr(rule, "rpn_program", "")),
        domain=str(getattr(rule, "domain", "text")),
        symbol_refs=_coerce_symbol_refs(getattr(rule, "symbol_refs", []) or []),
        word_refs=[str(value) for value in list(getattr(rule, "word_refs", []) or [])],
        examples=[
            dict(example) if isinstance(example, dict) else str(example)
            for example in list(getattr(rule, "examples", []) or [])
            if isinstance(example, dict) or str(example).strip()
        ],
        description=getattr(rule, "description", None),
        semantics=dict(semantics) if isinstance(semantics, dict) else {},
        usage_conditions=[
            str(value)
            for value in list(getattr(rule, "usage_conditions", []) or [])
            if str(value).strip()
        ],
        is_canonical=bool(getattr(rule, "is_canonical", False)),
        rule_strength=_clamp_rule_strength(
            getattr(rule, "rule_strength", 0),
            rule_id=str(getattr(rule, "rule_id", "")),
        ),
        superior_to=[
            str(value)
            for value in list(getattr(rule, "superior_to", []) or [])
            if str(value).strip()
        ],
        trust_weight=max(
            0.0,
            min(
                1.0,
                float(getattr(rule, "trust_weight", 1.0) or 1.0),
            ),
        ),
    )


def default_grammar_rules() -> list[GrammarRule]:
    return [_upgrade_rule(rule) for rule in _LEGACY_MODULE.default_grammar_rules()]


class GrammarGalaxy(_LEGACY_MODULE.GrammarGalaxy):
    """Legacy GrammarGalaxy with defeasible-rule metadata support."""

    def __init__(
        self,
        rules: list[GrammarRule] | None = None,
        users: dict[str, dict[str, Any]] | None = None,
        variants: dict[str, dict[str, str]] | None = None,
        storage_path: Path | None = None,
        snapshot: bytes | None = None,
    ):
        upgraded_rules = [_upgrade_rule(rule) for rule in (rules or default_grammar_rules())]
        super().__init__(
            rules=upgraded_rules,
            users=users,
            variants=variants,
            storage_path=storage_path,
            snapshot=snapshot,
        )
        self.rules = {
            str(rule_id): _upgrade_rule(rule)
            for rule_id, rule in dict(getattr(self, "rules", {})).items()
        }

    def add_rule(self, rule: GrammarRule, persist: bool = True) -> bool:
        return super().add_rule(_upgrade_rule(rule), persist=persist)

    def save(self, path: Path) -> None:
        rules_data = []
        for rule in self.rules.values():
            rule = _upgrade_rule(rule)
            rules_data.append(
                {
                    "rule_id": rule.rule_id,
                    "language": rule.language,
                    "pattern": rule.pattern,
                    "domain": rule.domain,
                    "rpn_program": rule.rpn_program,
                    "examples": [
                        dict(example) if isinstance(example, dict) else str(example)
                        for example in list(rule.examples)
                    ],
                    "description": rule.description,
                    "semantics": dict(rule.semantics),
                    "usage_conditions": list(rule.usage_conditions),
                    "is_canonical": bool(rule.is_canonical),
                    "symbol_refs": list(rule.symbol_refs),
                    "word_refs": list(rule.word_refs),
                    "rule_strength": int(rule.rule_strength),
                    "superior_to": list(rule.superior_to),
                    "trust_weight": float(rule.trust_weight),
                }
            )
        parameters = getattr(self, "_parameters", {})
        state = {"rules": rules_data, "total_count": len(rules_data), "parameters": parameters}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        state = json.loads(path.read_text(encoding="utf-8"))
        loaded_rules: dict[str, GrammarRule] = {}
        for row in state.get("rules", []):
            if not isinstance(row, dict):
                continue
            rule = GrammarRule(
                rule_id=str(row.get("rule_id", "")),
                language=str(row.get("language", "en")),
                pattern=str(row.get("pattern", "unknown")),
                rpn_program=str(row.get("rpn_program", "")),
                domain=str(row.get("domain", "general")),
                symbol_refs=_coerce_symbol_refs(row.get("symbol_refs", []) or []),
                word_refs=[str(value) for value in list(row.get("word_refs", []) or [])],
                examples=[
                    dict(example) if isinstance(example, dict) else str(example)
                    for example in list(row.get("examples", []) or [])
                    if isinstance(example, dict) or str(example).strip()
                ],
                description=row.get("description"),
                semantics=dict(row.get("semantics", {}) or {}),
                usage_conditions=[
                    str(value)
                    for value in list(row.get("usage_conditions", []) or [])
                    if str(value).strip()
                ],
                is_canonical=bool(row.get("is_canonical", False)),
                rule_strength=_clamp_rule_strength(
                    row.get("rule_strength", 0),
                    rule_id=str(row.get("rule_id", "")),
                ),
                superior_to=[
                    str(value)
                    for value in list(row.get("superior_to", []) or [])
                    if str(value).strip()
                ],
                trust_weight=max(0.0, min(1.0, float(row.get("trust_weight", 1.0) or 1.0))),
            )
            loaded_rules[rule.rule_id] = rule
        if loaded_rules:
            self.rules = loaded_rules
        parameters = state.get("parameters", {})
        if parameters:
            setattr(self, "_parameters", parameters)


def get_grammar_galaxy() -> GrammarGalaxy:
    return GrammarGalaxy()


__all__ = [
    "GrammarRule",
    "GrammarGalaxy",
    "default_grammar_rules",
    "get_grammar_galaxy",
]
