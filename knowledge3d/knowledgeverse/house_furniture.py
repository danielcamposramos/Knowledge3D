"""House furniture templates expressed as executable mesh construction stars."""

from __future__ import annotations

from ._house_utils import surface_forms
from .meaning_star import MeaningCentricStar


HOUSE_FURNITURE: list[MeaningCentricStar] = [
    MeaningCentricStar(
        star_id="furniture_sofa",
        meaning_class="furniture",
        meaning_rpn="SOFA SEATING COMFORT COLLABORATION SHARED DOMAIN_CENTER",
        domain="House/LivingRoom/Furniture",
        visual_rpn=(
            "1.0 GEN_CUBE 2.4 0.45 1.0 MAT4_SCALE MAT4_APPLY "
            "0.0 0.3 0.0 MAT4_TRANSLATE MAT4_APPLY "
            "1.0 GEN_CUBE 2.4 0.7 0.2 MAT4_SCALE MAT4_APPLY "
            "0.0 0.8 -0.5 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "1.0 GEN_CUBE 0.15 0.55 0.9 MAT4_SCALE MAT4_APPLY "
            "-1.2 0.55 -0.05 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "1.0 GEN_CUBE 0.15 0.55 0.9 MAT4_SCALE MAT4_APPLY "
            "1.2 0.55 -0.05 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
        ),
        behavior_rpn="SUPPORT SEATED COLLABORATE",
        surface_forms=surface_forms("sofa", "sofa", "ソファ"),
        house_position=(0.0, 0.0, -2.5),
        house_room="House/LivingRoom",
        confidence=1,
        polarity=1,
        taxonomy_refs=["concept_visual_art", "concept_language"],
    ),
    MeaningCentricStar(
        star_id="furniture_holodesk",
        meaning_class="furniture",
        meaning_rpn="HOLODESK PROJECTION SURFACE COLLABORATION 3D AUGMENTED DOMAIN_CENTER",
        domain="House/LivingRoom/Furniture",
        visual_rpn=(
            "1.0 GEN_CUBE 1.6 0.04 0.9 MAT4_SCALE MAT4_APPLY "
            "0.0 0.42 0.0 MAT4_TRANSLATE MAT4_APPLY "
            "0.04 0.40 8 1 GEN_CYLINDER -0.7 0.20 -0.38 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "0.04 0.40 8 1 GEN_CYLINDER 0.7 0.20 -0.38 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "0.04 0.40 8 1 GEN_CYLINDER -0.7 0.20 0.38 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "0.04 0.40 8 1 GEN_CYLINDER 0.7 0.20 0.38 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "1.0 GEN_CUBE 1.68 0.02 0.98 MAT4_SCALE MAT4_APPLY "
            "1.0 GEN_CUBE 1.56 0.04 0.86 MAT4_SCALE MAT4_APPLY CSG_SUBTRACT "
            "0.0 0.45 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
        ),
        behavior_rpn="HOLODESK ACTIVATE PROJECT_3D COLLABORATE SHARE_MODELS",
        surface_forms=surface_forms("HoloDesk", "HoloMesa", "ホロデスク"),
        house_position=(0.0, 0.0, 0.0),
        house_room="House/LivingRoom",
        confidence=1,
        polarity=1,
        taxonomy_refs=[
            "concept_visual_art",
            "concept_tool",
            "concept_mathematics",
            "concept_language",
        ],
    ),
    MeaningCentricStar(
        star_id="furniture_bookshelf",
        meaning_class="furniture",
        meaning_rpn="SHELF STORAGE BOOKS VERTICAL DOMAIN_CENTER",
        domain="House/Library/Furniture",
        visual_rpn=(
            "1.0 GEN_CUBE 2.0 1.8 0.08 MAT4_SCALE MAT4_APPLY "
            "1.0 GEN_CUBE 2.0 0.08 0.8 MAT4_SCALE MAT4_APPLY 0.0 -0.75 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "1.0 GEN_CUBE 2.0 0.08 0.8 MAT4_SCALE MAT4_APPLY 0.0 -0.35 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "1.0 GEN_CUBE 2.0 0.08 0.8 MAT4_SCALE MAT4_APPLY 0.0 0.05 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "1.0 GEN_CUBE 2.0 0.08 0.8 MAT4_SCALE MAT4_APPLY 0.0 0.45 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "1.0 GEN_CUBE 2.0 0.08 0.8 MAT4_SCALE MAT4_APPLY 0.0 0.85 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "1.0 GEN_CUBE 0.08 1.8 0.8 MAT4_SCALE MAT4_APPLY -1.0 0.0 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "1.0 GEN_CUBE 0.08 1.8 0.8 MAT4_SCALE MAT4_APPLY 1.0 0.0 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
        ),
        behavior_rpn="PLACE BOOKS ALIGN SHELVES",
        surface_forms=surface_forms("bookshelf", "estante", "本棚"),
        house_position=(0.0, 0.0, 0.0),
        house_room="House/Library",
        confidence=1,
        polarity=1,
        taxonomy_refs=["concept_language", "concept_mathematics"],
    ),
    MeaningCentricStar(
        star_id="furniture_desk",
        meaning_class="furniture",
        meaning_rpn="DESK SURFACE WORK HORIZONTAL DOMAIN_CENTER",
        domain="House/Library/Furniture",
        visual_rpn=(
            "1.0 GEN_CUBE 1.5 0.08 0.8 MAT4_SCALE MAT4_APPLY 0.0 0.75 0.0 MAT4_TRANSLATE MAT4_APPLY "
            "0.03 0.75 8 1 GEN_CYLINDER -0.7 0.375 -0.35 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "0.03 0.75 8 1 GEN_CYLINDER 0.7 0.375 -0.35 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "0.03 0.75 8 1 GEN_CYLINDER -0.7 0.375 0.35 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "0.03 0.75 8 1 GEN_CYLINDER 0.7 0.375 0.35 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
        ),
        behavior_rpn="WORK SURFACE READY",
        surface_forms=surface_forms("desk", "mesa", "机"),
        house_position=(3.0, 0.0, 2.0),
        house_room="House/Library",
        confidence=1,
        polarity=1,
        taxonomy_refs=["concept_tool", "concept_language"],
    ),
    MeaningCentricStar(
        star_id="furniture_chair",
        meaning_class="furniture",
        meaning_rpn="CHAIR SUPPORT REST SEAT DOMAIN_CENTER",
        domain="House/Library/Furniture",
        visual_rpn=(
            "1.0 GEN_CUBE 0.75 0.08 0.75 MAT4_SCALE MAT4_APPLY 0.0 0.55 0.0 MAT4_TRANSLATE MAT4_APPLY "
            "1.0 GEN_CUBE 0.75 0.8 0.08 MAT4_SCALE MAT4_APPLY 0.0 1.0 -0.34 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "0.03 0.55 8 1 GEN_CYLINDER -0.3 0.275 -0.3 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "0.03 0.55 8 1 GEN_CYLINDER 0.3 0.275 -0.3 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "0.03 0.55 8 1 GEN_CYLINDER -0.3 0.275 0.3 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "0.03 0.55 8 1 GEN_CYLINDER 0.3 0.275 0.3 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
        ),
        behavior_rpn="SUPPORT SEATED READING",
        surface_forms=surface_forms("chair", "cadeira", "椅子"),
        house_position=(2.0, 0.0, 2.8),
        house_room="House/Library",
        confidence=1,
        polarity=1,
        taxonomy_refs=["concept_language"],
    ),
    MeaningCentricStar(
        star_id="furniture_workbench",
        meaning_class="furniture",
        meaning_rpn="WORKBENCH TOOLS CONSTRUCTION APPLY DOMAIN_CENTER",
        domain="House/Workshop/Furniture",
        visual_rpn=(
            "1.0 GEN_CUBE 2.2 0.1 1.0 MAT4_SCALE MAT4_APPLY 0.0 0.95 0.0 MAT4_TRANSLATE MAT4_APPLY "
            "1.0 GEN_CUBE 2.0 0.1 0.25 MAT4_SCALE MAT4_APPLY 0.0 1.45 -0.35 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "0.05 0.95 10 1 GEN_CYLINDER -0.95 0.475 -0.4 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "0.05 0.95 10 1 GEN_CYLINDER 0.95 0.475 -0.4 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "0.05 0.95 10 1 GEN_CYLINDER -0.95 0.475 0.4 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "0.05 0.95 10 1 GEN_CYLINDER 0.95 0.475 0.4 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
        ),
        behavior_rpn="TOOLS READY BUILD MODE",
        surface_forms=surface_forms("workbench", "bancada", "作業台"),
        house_position=(0.0, 0.0, 0.0),
        house_room="House/Workshop",
        confidence=1,
        polarity=1,
        taxonomy_refs=["concept_tool", "concept_mathematics"],
    ),
    MeaningCentricStar(
        star_id="furniture_bathtub",
        meaning_class="furniture",
        meaning_rpn="BATHTUB PORTAL INTROSPECTION VESSEL DOMAIN_CENTER",
        domain="House/Bathtub/Furniture",
        visual_rpn=(
            "0.6 0.0 MOVE 0.72 0.1 LINE 0.72 0.4 LINE "
            "0.48 0.58 0.24 0.48 QUAD 0.0 0.48 LINE CLOSE 20 LATHE"
        ),
        behavior_rpn="PORTAL REST REFLECT",
        surface_forms=surface_forms("bathtub", "banheira", "浴槽"),
        house_position=(0.0, 0.0, 0.0),
        house_room="House/Bathtub",
        confidence=1,
        polarity=1,
        taxonomy_refs=["concept_self_reflection", "concept_sound"],
    ),
    MeaningCentricStar(
        star_id="furniture_knowledge_tree",
        meaning_class="furniture",
        meaning_rpn="TREE ONTOLOGY BRANCHES LEAVES GROWTH DOMAIN_CENTER",
        domain="House/Garden/Furniture",
        visual_rpn=(
            "0.15 2.0 8 1 GEN_CYLINDER "
            "1.5 1 GEN_ICOSPHERE 0.0 2.5 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
        ),
        behavior_rpn="GROW BRANCH LINK KNOWLEDGE",
        surface_forms=surface_forms("knowledge tree", "arvore do conhecimento", "知識の木"),
        house_position=(5.0, 0.0, 5.0),
        house_room="House/Garden",
        confidence=1,
        polarity=1,
        taxonomy_refs=["concept_growth", "concept_language", "concept_biology"],
        component_refs=[
            "tree_branch_mathematics",
            "tree_branch_language",
            "tree_branch_physics",
            "tree_branch_biology",
            "tree_branch_tools",
        ],
    ),
]


__all__ = ["HOUSE_FURNITURE"]
