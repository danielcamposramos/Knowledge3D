"""Physical House instantiation of the Memory Tablet interface."""

from __future__ import annotations

from ._house_utils import surface_forms
from .meaning_star import MeaningCentricStar


MEMORY_TABLET = MeaningCentricStar(
    star_id="memory_tablet",
    meaning_class="tablet",
    meaning_rpn="TABLET MEMORY INTERFACE BROWSE QUERY INSPECT DOMAIN_CENTER",
    domain="House/Interface",
    taxonomy_refs=[
        "concept_language",
        "concept_mathematics",
        "concept_visual_art",
        "concept_physics",
        "concept_biology",
        "concept_tool",
    ],
    surface_forms=surface_forms("Memory Tablet", "Tablete de Memoria", "記憶タブレット"),
    visual_rpn=(
        "1.0 GEN_CUBE 0.40 0.28 0.02 MAT4_SCALE MAT4_APPLY "
        "1.0 GEN_CUBE 0.36 0.24 0.01 MAT4_SCALE MAT4_APPLY "
        "0.0 0.0 0.011 MAT4_TRANSLATE MAT4_APPLY CSG_SUBTRACT"
    ),
    behavior_rpn="TABLET ACTIVATE BROWSE_GALAXY QUERY_KNOWLEDGE INSPECT_PROGRAMS",
    house_position=(0.0, 1.0, 2.0),
    house_room="House",
    confidence=1,
    polarity=1,
)


__all__ = ["MEMORY_TABLET"]
