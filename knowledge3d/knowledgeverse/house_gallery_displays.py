"""Gallery display-frame stars for curated House wall exhibits."""

from __future__ import annotations

import math

from ._house_utils import surface_forms
from .meaning_star import MeaningCentricStar


def _display_frame_visual(width: float, height: float, depth: float, rotation_y: float = 0.0) -> str:
    inner_width = max(width - 0.06, 0.04)
    inner_height = max(height - 0.06, 0.04)
    visual = (
        f"1.0 GEN_CUBE {width:.2f} {height:.2f} {depth:.2f} MAT4_SCALE MAT4_APPLY "
        f"1.0 GEN_CUBE {inner_width:.2f} {inner_height:.2f} {depth + 0.01:.2f} MAT4_SCALE MAT4_APPLY CSG_SUBTRACT"
    )
    if rotation_y:
        visual += f" {rotation_y:.4f} MAT4_ROTATE_Y MAT4_APPLY"
    return visual


def _display(
    *,
    star_id: str,
    title_en: str,
    title_pt: str,
    title_ja: str,
    meaning_rpn: str,
    visual_rpn: str,
    taxonomy_refs: list[str],
    house_position: tuple[float, float, float],
) -> MeaningCentricStar:
    return MeaningCentricStar(
        star_id=star_id,
        meaning_class="display",
        meaning_rpn=meaning_rpn,
        domain="House/Gallery/Displays",
        taxonomy_refs=taxonomy_refs,
        surface_forms=surface_forms(title_en, title_pt, title_ja),
        visual_rpn=visual_rpn,
        behavior_rpn="DISPLAY CURATE REFERENCE_GALAXY",
        house_position=house_position,
        house_room="House/Gallery",
        confidence=1,
        polarity=1,
    )


GALLERY_DISPLAYS: list[MeaningCentricStar] = [
    _display(
        star_id="display_drawing_primitives",
        title_en="Drawing Primitives",
        title_pt="Primitivas de Desenho",
        title_ja="描画プリミティブ",
        meaning_rpn="DISPLAY DRAWING PRIMITIVES CURVES LINES GLYPHS",
        visual_rpn=_display_frame_visual(1.20, 0.90, 0.04, rotation_y=math.pi / 2.0),
        taxonomy_refs=["concept_visual_art", "glyph_curve_transfer", "cubic_bezier_eval"],
        house_position=(-6.65, 0.65, 0.0),
    ),
    _display(
        star_id="display_number_line",
        title_en="Number Line",
        title_pt="Reta Numerica",
        title_ja="数直線",
        meaning_rpn="DISPLAY NUMBER LINE COUNTING ORDER MAGNITUDE",
        visual_rpn=_display_frame_visual(1.35, 0.72, 0.04, rotation_y=0.0),
        taxonomy_refs=["concept_mathematics", "num_0", "num_1", "num_2", "num_10"],
        house_position=(0.0, 0.60, -6.30),
    ),
    _display(
        star_id="display_character_forms",
        title_en="Character Forms",
        title_pt="Formas de Caracteres",
        title_ja="文字の形",
        meaning_rpn="DISPLAY CHARACTER FORMS MULTILINGUAL GLYPHS LANGUAGE",
        visual_rpn=_display_frame_visual(0.90, 1.20, 0.04, rotation_y=math.pi / 2.0),
        taxonomy_refs=["concept_language", "seed_word_en_language", "seed_word_pt_linguagem", "seed_word_ja_gengo"],
        house_position=(6.65, 0.75, 0.0),
    ),
    _display(
        star_id="display_physics_forces",
        title_en="Physics Forces",
        title_pt="Forcas da Fisica",
        title_ja="物理の力",
        meaning_rpn="DISPLAY PHYSICS FORCES MOTION ENERGY DYNAMICS",
        visual_rpn=_display_frame_visual(1.10, 0.88, 0.04, rotation_y=math.pi),
        taxonomy_refs=[
            "concept_physics",
            "reality_dynamics_newton_second_law",
            "reality_dynamics_friction_force",
            "reality_dynamics_hooke_law",
        ],
        house_position=(0.0, 0.62, 6.30),
    ),
]


__all__ = ["GALLERY_DISPLAYS"]
