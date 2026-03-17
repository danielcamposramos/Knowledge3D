"""Foundational House room templates built from H1/H2 construction primitives."""

from __future__ import annotations

from ._house_utils import surface_forms
from .meaning_star import MeaningCentricStar


HOUSE_ROOMS: list[MeaningCentricStar] = [
    MeaningCentricStar(
        star_id="room_living",
        meaning_class="room",
        meaning_rpn="ROOM COLLABORATION SHARED PROJECTION DOMAIN_CENTER",
        domain="House/LivingRoom",
        visual_rpn=(
            "10.0 GEN_CUBE 9.4 GEN_CUBE CSG_SUBTRACT "
            "1.0 GEN_CUBE 1.3 1.6 0.35 MAT4_SCALE MAT4_APPLY -5.0 -1.0 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_SUBTRACT "
            "1.0 GEN_CUBE 1.3 1.6 0.35 MAT4_SCALE MAT4_APPLY 5.0 -1.0 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_SUBTRACT"
        ),
        behavior_rpn="ROOM_ENTER LOAD_KNOWLEDGE_DOMAIN COLLABORATION ACTIVATE_HOLODESK",
        surface_forms=surface_forms("living room", "sala de estar", "リビングルーム"),
        house_position=(-10.0, 0.0, 0.0),
        house_room="House/LivingRoom",
        confidence=1,
        polarity=1,
        taxonomy_refs=["concept_visual_art", "concept_tool", "concept_language"],
        component_refs=[
            "furniture_sofa",
            "furniture_holodesk",
        ],
    ),
    MeaningCentricStar(
        star_id="room_library",
        meaning_class="room",
        meaning_rpn="ROOM KNOWLEDGE BOOKS READING DOMAIN_CENTER",
        domain="House/Library",
        visual_rpn=(
            "8.0 GEN_CUBE 7.6 GEN_CUBE CSG_SUBTRACT "
            "1.0 GEN_CUBE 1.3 1.6 0.35 MAT4_SCALE MAT4_APPLY 4.0 -1.0 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_SUBTRACT "
            "1.0 GEN_CUBE 2.4 0.08 5.8 MAT4_SCALE MAT4_APPLY 0.0 1.6 -2.7 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
        ),
        behavior_rpn="ROOM_ENTER LOAD_KNOWLEDGE_DOMAIN LIBRARY ACTIVATE_SHELVES",
        surface_forms=surface_forms("library", "biblioteca", "図書館"),
        house_position=(0.0, 0.0, 0.0),
        house_room="House/Library",
        confidence=1,
        polarity=1,
        taxonomy_refs=["concept_language", "concept_mathematics", "concept_growth"],
        component_refs=[
            "furniture_bookshelf",
            "furniture_desk",
            "furniture_chair",
            "book_mathematics_primer",
            "book_language_foundations",
            "book_physics_handbook",
            "book_biology_atlas",
            "book_tool_manual",
        ],
    ),
    MeaningCentricStar(
        star_id="room_garden",
        meaning_class="room",
        meaning_rpn="ROOM GROWTH ONTOLOGY TREES EXPLORATION DOMAIN_CENTER",
        domain="House/Garden",
        visual_rpn=(
            "20.0 20.0 4 4 GEN_PLANE "
            "20.0 GEN_CUBE 18.8 GEN_CUBE CSG_SUBTRACT "
            "1.0 0.08 1.0 MAT4_SCALE MAT4_APPLY 0.0 0.55 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
        ),
        behavior_rpn="ROOM_ENTER LOAD_KNOWLEDGE_DOMAIN ONTOLOGY GROW_TREES",
        surface_forms=surface_forms("knowledge garden", "jardim do conhecimento", "知識の庭"),
        house_position=(18.0, 0.0, 0.0),
        house_room="House/Garden",
        confidence=1,
        polarity=1,
        taxonomy_refs=["concept_growth", "concept_biology", "concept_language"],
        component_refs=["furniture_knowledge_tree", "concept_growth"],
    ),
    MeaningCentricStar(
        star_id="room_workshop",
        meaning_class="room",
        meaning_rpn="ROOM TOOLS CONSTRUCTION BUILD APPLY DOMAIN_CENTER",
        domain="House/Workshop",
        visual_rpn=(
            "10.0 GEN_CUBE 9.4 GEN_CUBE CSG_SUBTRACT "
            "1.0 GEN_CUBE 2.6 1.4 0.35 MAT4_SCALE MAT4_APPLY 5.0 -1.0 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_SUBTRACT "
            "1.0 GEN_CUBE 3.0 0.12 1.2 MAT4_SCALE MAT4_APPLY 0.0 -1.2 -3.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
        ),
        behavior_rpn="ROOM_ENTER LOAD_KNOWLEDGE_DOMAIN WORKSHOP ACTIVATE_TOOLS",
        surface_forms=surface_forms("workshop", "oficina", "工房"),
        house_position=(36.0, 0.0, 0.0),
        house_room="House/Workshop",
        confidence=1,
        polarity=1,
        taxonomy_refs=["concept_tool", "concept_mathematics", "concept_visual_art"],
        component_refs=[
            "furniture_workbench",
            "concept_tool",
            "tool_obj_hammer",
            "tool_obj_wrench",
            "tool_obj_brush",
            "tool_obj_tuning_fork",
            "tool_obj_lens",
        ],
    ),
    MeaningCentricStar(
        star_id="room_gallery",
        meaning_class="room",
        meaning_rpn="ROOM VISUAL AUDIO ART PERCEPTION DOMAIN_CENTER",
        domain="House/Gallery",
        visual_rpn=(
            "14.0 GEN_CUBE 13.4 GEN_CUBE CSG_SUBTRACT "
            "1.0 GEN_CUBE 3.0 1.6 0.35 MAT4_SCALE MAT4_APPLY 7.0 -0.8 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_SUBTRACT "
            "1.0 GEN_CUBE 0.25 2.4 4.0 MAT4_SCALE MAT4_APPLY -5.5 0.0 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
        ),
        behavior_rpn="ROOM_ENTER LOAD_KNOWLEDGE_DOMAIN GALLERY ACTIVATE_DISPLAY",
        surface_forms=surface_forms("gallery", "galeria", "ギャラリー"),
        house_position=(54.0, 0.0, 0.0),
        house_room="House/Gallery",
        confidence=1,
        polarity=1,
        taxonomy_refs=["concept_visual_art", "concept_sound", "concept_language"],
        component_refs=[
            "concept_visual_art",
            "concept_sound",
            "display_drawing_primitives",
            "display_number_line",
            "display_character_forms",
            "display_physics_forces",
        ],
    ),
    MeaningCentricStar(
        star_id="room_bathtub",
        meaning_class="room",
        meaning_rpn="ROOM INTROSPECTION GALAXY META OBSERVE DOMAIN_CENTER",
        domain="House/Bathtub",
        visual_rpn=(
            "6.0 GEN_CUBE 5.6 GEN_CUBE CSG_SUBTRACT "
            "0.8 0.0 MOVE 0.92 0.16 LINE 0.92 0.52 LINE "
            "0.56 0.7 0.22 0.6 QUAD 0.0 0.6 LINE CLOSE 20 LATHE "
            "0.0 -1.2 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
        ),
        behavior_rpn="ROOM_ENTER INTROSPECT_MODE ACTIVATE_GALAXY_VIEW",
        surface_forms=surface_forms("bathtub observatory", "banheira observatorio", "浴槽観測室"),
        house_position=(72.0, 0.0, 0.0),
        house_room="House/Bathtub",
        confidence=1,
        polarity=1,
        taxonomy_refs=["concept_self_reflection", "concept_sound", "concept_growth"],
        component_refs=[
            "furniture_bathtub",
            "concept_self_reflection",
            "observatory_telescope",
            "observatory_prism",
            "observatory_journal",
        ],
    ),
]


__all__ = ["HOUSE_ROOMS"]
