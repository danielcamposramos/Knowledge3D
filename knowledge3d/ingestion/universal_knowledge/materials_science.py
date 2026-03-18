"""Material-composition rules referencing elemental building blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class MaterialRule:
    key: str
    name: str
    formula: str
    rule_rpn: str
    category: str
    element_symbols: tuple[str, ...] = field(default_factory=tuple)
    component_refs: tuple[str, ...] = field(default_factory=tuple)


MATERIAL_RULES: dict[str, MaterialRule] = {
    "water_composition": MaterialRule(
        key="water_composition",
        name="water",
        formula="H2O",
        rule_rpn="ELEMENT_H 2 BOND_COVALENT ELEMENT_O 1 BOND_COVALENT MOLECULE_COMPOSE",
        category="compound",
        element_symbols=("H", "O"),
    ),
    "steel_composition": MaterialRule(
        key="steel_composition",
        name="steel",
        formula="Fe + C (0.2-2.1%)",
        rule_rpn="ELEMENT_FE ELEMENT_C 0.02 ALLOY_MIX",
        category="alloy",
        element_symbols=("Fe", "C"),
    ),
    "glass_composition": MaterialRule(
        key="glass_composition",
        name="glass",
        formula="SiO2 + Na2O + CaO",
        rule_rpn="ELEMENT_SI ELEMENT_O 2 BOND_COVALENT ELEMENT_NA 2 ELEMENT_O FLUX_ADD ELEMENT_CA ELEMENT_O STABILIZE",
        category="amorphous_solid",
        element_symbols=("Si", "O", "Na", "Ca"),
    ),
    "wood_composition": MaterialRule(
        key="wood_composition",
        name="wood",
        formula="cellulose + hemicellulose + lignin",
        rule_rpn="CELLULOSE HEMICELLULOSE LIGNIN NATURAL_COMPOSITE",
        category="natural_composite",
        component_refs=("biology_plant",),
    ),
    "concrete_composition": MaterialRule(
        key="concrete_composition",
        name="concrete",
        formula="cement + water + aggregate",
        rule_rpn="CEMENT WATER 0.45 RATIO_MIX AGGREGATE ADD HYDRATION_CURE",
        category="composite",
        element_symbols=("Ca", "Si", "O", "H"),
        component_refs=("material_sand", "material_gravel"),
    ),
}


def iter_material_rules() -> list[MaterialRule]:
    return [MATERIAL_RULES[key] for key in sorted(MATERIAL_RULES.keys())]


def validate_material_rules(valid_symbols: Iterable[str]) -> bool:
    symbol_set = {str(symbol).strip() for symbol in valid_symbols if str(symbol).strip()}
    for rule in MATERIAL_RULES.values():
        for symbol in rule.element_symbols:
            if symbol not in symbol_set:
                return False
    return True


__all__ = ["MATERIAL_RULES", "MaterialRule", "iter_material_rules", "validate_material_rules"]
