"""Universal foundational knowledge registries for ingestion and House/Galaxy population."""

from __future__ import annotations

from typing import Iterable

from knowledge3d.knowledgeverse._house_utils import char_refs
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar, SurfaceForm

from .file_formats import FILE_FORMATS, iter_format_entries
from .materials_science import MATERIAL_RULES, MaterialRule, iter_material_rules, validate_material_rules
from .measurements import MEASUREMENT_DOMAINS, MeasurementDomain, UnitDefinition, convert, from_si_value, iter_domains, to_si_value
from .numeral_systems import NUMERAL_SYSTEMS, NumeralSystem, decode_number, encode_number, iter_numeral_systems
from .paper_and_book_sizes import BOOK_SIZES, PAPER_SIZES, StandardSize, a_series_ratio_ok, iter_book_sizes, iter_paper_sizes
from .periodic_table import ELEMENTS, ELEMENTS_BY_ATOMIC_NUMBER, ELEMENTS_BY_SYMBOL, ElementEntry, SUBATOMIC_PARTICLES, iter_elements
from .physical_constants import PHYSICAL_CONSTANTS, PhysicalConstant, iter_physical_constants
from .proceduralize import proceduralize_content, proceduralize_text, tokenize_words
from .wikipedia_ingest import WikipediaAttribution, WikipediaIngestRecord, build_wikipedia_record
from .writing_systems import WRITING_SYSTEMS, WritingSystem, get_writing_system, iter_writing_systems


def _surface_map(values: dict[str, str]) -> dict[str, SurfaceForm]:
    out: dict[str, SurfaceForm] = {}
    for language, text in dict(values or {}).items():
        lang = str(language).strip().lower()
        raw = str(text).strip()
        if not lang or not raw:
            continue
        out[lang] = SurfaceForm(
            word_ref=f"{lang}_{raw.lower().replace(' ', '_')}",
            char_refs=char_refs(raw, lang),
        )
    if "en" not in out:
        out["en"] = SurfaceForm(word_ref="en_entry", char_refs=["char_e"])
    return out


def build_foundation_stars(
    *,
    include_elements: bool = True,
    include_units: bool = True,
) -> list[MeaningCentricStar]:
    """Build a compact meaning-star layer from the universal registries.

    This intentionally emits category- and entry-level stars rather than one star
    per Unicode character, which would be too large for the default static export.
    """
    stars: list[MeaningCentricStar] = []

    for system in iter_writing_systems():
        stars.append(
            MeaningCentricStar(
                star_id=f"script_{system.key}",
                meaning_class="script",
                meaning_rpn=f"SCRIPT {system.name.upper().replace(' ', '_')}",
                domain="Foundation/WritingSystems",
                taxonomy_refs=["concept_language", "standard_unicode_script"],
                surface_forms=_surface_map({"en": system.name}),
                meta_refs=[f"unicode_range:{system.unicode_start:04X}-{system.unicode_end:04X}"],
                house_room="House/Library",
                confidence=1,
                polarity=1,
            )
        )

    for system in iter_numeral_systems():
        stars.append(
            MeaningCentricStar(
                star_id=f"numeral_system_{system.key}",
                meaning_class="numeral_system",
                meaning_rpn=f"NUMERAL_SYSTEM {system.name.upper().replace(' ', '_')}",
                domain="Foundation/Numerals",
                taxonomy_refs=["concept_mathematics"],
                surface_forms=_surface_map(system.surface_forms or {"en": system.name}),
                grammar_refs=[f"grammar_numeral_{system.key}"],
                house_room="House/Library",
                confidence=1,
                polarity=1,
            )
        )

    for size in list(iter_paper_sizes()) + list(iter_book_sizes()):
        stars.append(
            MeaningCentricStar(
                star_id=f"standard_size_{size.key}",
                meaning_class="standard_size",
                meaning_rpn=f"SIZE {size.width_mm:.3f}MM {size.height_mm:.3f}MM",
                domain="Foundation/Standards",
                taxonomy_refs=["concept_tool", "standard_size"],
                surface_forms=_surface_map({"en": size.label}),
                meta_refs=[f"standard:{size.standard}", f"category:{size.category}"],
                house_room="House/Workshop",
                confidence=1,
                polarity=1,
            )
        )

    for extension, entry in iter_format_entries():
        stars.append(
            MeaningCentricStar(
                star_id=f"format_{extension}",
                meaning_class="file_format",
                meaning_rpn=f"FORMAT {extension.upper()} MIME {entry['mime'].upper().replace('/', '_')}",
                domain=f"Foundation/{entry['domain']}",
                taxonomy_refs=["concept_tool", "standard_open_format"],
                surface_forms=_surface_map({"en": extension}),
                meta_refs=[entry["mime"], entry["description"]],
                house_room="House/Workshop",
                confidence=1,
                polarity=1,
            )
        )

    if include_units:
        for domain in iter_domains():
            for unit_name, unit in domain.units.items():
                stars.append(
                    MeaningCentricStar(
                        star_id=f"unit_{domain.key}_{unit_name}",
                        meaning_class="measurement_unit",
                        meaning_rpn=unit.to_si_rpn,
                        domain=f"Foundation/{domain.key.title()}",
                        taxonomy_refs=["concept_mathematics", f"unit_domain_{domain.key}"],
                        surface_forms=_surface_map({"en": unit_name.replace('_', ' ')}),
                        meta_refs=[f"symbol:{unit.symbol}", f"si_base:{domain.si_base}"],
                        house_room="House/Library",
                        confidence=1,
                        polarity=1,
                    )
                )

    for constant in iter_physical_constants():
        stars.append(
            MeaningCentricStar(
                star_id=f"constant_{constant.key}",
                meaning_class="physical_constant",
                meaning_rpn=f"CONSTANT {constant.symbol} {constant.value}",
                domain="Foundation/Physics",
                taxonomy_refs=["concept_physics", "physical_constant"],
                surface_forms=_surface_map({"en": constant.name}),
                meta_refs=[f"unit:{constant.unit}", f"exact:{int(constant.exact)}"],
                house_room="House/Library",
                confidence=1,
                polarity=1,
            )
        )

    if include_elements:
        for element in iter_elements():
            stars.append(
                MeaningCentricStar(
                    star_id=f"element_{element.symbol.lower()}",
                    meaning_class="chemical_element",
                    meaning_rpn=f"ELEMENT {element.symbol} ATOMIC_NUMBER {element.atomic_number}",
                    domain="Foundation/Chemistry",
                    taxonomy_refs=["concept_chemistry", "periodic_table"],
                    surface_forms=_surface_map(element.surface_forms),
                    meta_refs=[
                        f"group:{element.group if element.group is not None else 'none'}",
                        f"period:{element.period}",
                        f"block:{element.block}",
                    ],
                    house_room="House/Library",
                    confidence=1,
                    polarity=1,
                )
            )

    for material in iter_material_rules():
        stars.append(
            MeaningCentricStar(
                star_id=f"material_{material.key}",
                meaning_class="material_rule",
                meaning_rpn=material.rule_rpn,
                domain="Foundation/Materials",
                taxonomy_refs=["concept_chemistry", "materials_science"],
                surface_forms=_surface_map({"en": material.name}),
                reality_refs=list(material.element_symbols),
                house_room="House/Workshop",
                confidence=1,
                polarity=1,
            )
        )

    return stars


__all__ = [
    "BOOK_SIZES",
    "ELEMENTS",
    "ELEMENTS_BY_ATOMIC_NUMBER",
    "ELEMENTS_BY_SYMBOL",
    "ElementEntry",
    "FILE_FORMATS",
    "MATERIAL_RULES",
    "MEASUREMENT_DOMAINS",
    "MeasurementDomain",
    "NUMERAL_SYSTEMS",
    "NumeralSystem",
    "PAPER_SIZES",
    "PHYSICAL_CONSTANTS",
    "PhysicalConstant",
    "SUBATOMIC_PARTICLES",
    "StandardSize",
    "UnitDefinition",
    "WikipediaAttribution",
    "WikipediaIngestRecord",
    "WRITING_SYSTEMS",
    "WritingSystem",
    "a_series_ratio_ok",
    "build_foundation_stars",
    "build_wikipedia_record",
    "convert",
    "decode_number",
    "encode_number",
    "from_si_value",
    "get_writing_system",
    "iter_book_sizes",
    "iter_domains",
    "iter_elements",
    "iter_format_entries",
    "iter_material_rules",
    "iter_numeral_systems",
    "iter_paper_sizes",
    "iter_physical_constants",
    "iter_writing_systems",
    "proceduralize_content",
    "proceduralize_text",
    "to_si_value",
    "tokenize_words",
    "validate_material_rules",
]
