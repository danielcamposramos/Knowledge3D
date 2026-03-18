from __future__ import annotations

from knowledge3d.ingestion.universal_knowledge import (
    ELEMENTS,
    ELEMENTS_BY_SYMBOL,
    MATERIAL_RULES,
    NUMERAL_SYSTEMS,
    PHYSICAL_CONSTANTS,
    WRITING_SYSTEMS,
    a_series_ratio_ok,
    convert,
    decode_number,
    encode_number,
    proceduralize_content,
    validate_material_rules,
)
from knowledge3d.tools.augmentation_providers import AugmentationResult


def test_all_writing_systems_have_unicode_range() -> None:
    for script in WRITING_SYSTEMS.values():
        assert script.unicode_start < script.unicode_end
        assert script.approx_chars > 0
        assert script.direction in {"LTR", "RTL", "TTB"}


def test_numeral_systems_round_trip() -> None:
    for system in NUMERAL_SYSTEMS.values():
        if not system.supports_roundtrip:
            continue
        encoded = encode_number(system.key, 42)
        assert decode_number(system.key, encoded) == 42


def test_paper_sizes_a_series_ratio() -> None:
    assert a_series_ratio_ok() is True


def test_unit_conversion_round_trip() -> None:
    kelvin = convert("temperature", 100.0, "fahrenheit", "kelvin")
    celsius = convert("temperature", kelvin, "kelvin", "celsius")
    fahrenheit = convert("temperature", kelvin, "kelvin", "fahrenheit")
    assert abs(celsius - 37.7777777778) < 1e-3
    assert abs(fahrenheit - 100.0) < 1e-3


def test_periodic_table_complete() -> None:
    assert len(ELEMENTS) == 118
    assert ELEMENTS[0].symbol == "H"
    assert ELEMENTS[-1].symbol == "Og"
    assert ELEMENTS_BY_SYMBOL["Fe"].atomic_number == 26


def test_periodic_table_groups() -> None:
    carbon = ELEMENTS_BY_SYMBOL["C"]
    iron = ELEMENTS_BY_SYMBOL["Fe"]
    uranium = ELEMENTS_BY_SYMBOL["U"]
    assert (carbon.group, carbon.period, carbon.block) == (14, 2, "p")
    assert (iron.group, iron.period, iron.block) == (8, 4, "d")
    assert (uranium.group, uranium.period, uranium.block) == (None, 7, "f")


def test_material_composition_references_elements() -> None:
    assert validate_material_rules(ELEMENTS_BY_SYMBOL.keys()) is True
    assert "water_composition" in MATERIAL_RULES


def test_proceduralize_creates_symlinks() -> None:
    result = AugmentationResult(
        summary="Water supports life",
        entities=[{"type": "concept", "name": "water", "content": "H2O"}],
        relationships=[],
        domain="Biology",
        meaning_rpn_hint="BIOLOGY WATER ENTRY",
        taxonomy_refs=["concept_biology"],
        surface_forms={"en": "Water", "pt": "Agua"},
        confidence=0.9,
        provider="test",
        raw_response="{}",
    )

    def _lookup(token: str) -> str | None:
        return {
            "water": "concept_water",
            "supports": "rel_supports",
            "life": "concept_life",
        }.get(token)

    records = proceduralize_content(result, galaxy_lookup=_lookup)
    summary_links = records[0]["links"]
    assert summary_links[0]["star_id"] == "concept_water"
    assert summary_links[1]["star_id"] == "rel_supports"
    assert summary_links[2]["star_id"] == "concept_life"


def test_physical_constants_exact_values() -> None:
    assert PHYSICAL_CONSTANTS["speed_of_light"].exact is True
    assert PHYSICAL_CONSTANTS["speed_of_light"].value == 2.99792458e8
    assert PHYSICAL_CONSTANTS["planck_constant"].value == 6.62607015e-34
    assert PHYSICAL_CONSTANTS["elementary_charge"].value == 1.602176634e-19
