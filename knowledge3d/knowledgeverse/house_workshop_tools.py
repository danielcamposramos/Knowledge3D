"""Workshop tool-object stars placed on the House workbench."""

from __future__ import annotations

from ._house_utils import surface_forms
from .meaning_star import MeaningCentricStar


def _tool_object(
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
        meaning_class="tool_object",
        meaning_rpn=meaning_rpn,
        domain="House/Workshop/Tools",
        taxonomy_refs=taxonomy_refs,
        surface_forms=surface_forms(title_en, title_pt, title_ja),
        visual_rpn=visual_rpn,
        behavior_rpn="TOOL_OBJECT INSPECT MAP_TO_TOOL_GALAXY",
        house_position=house_position,
        house_room="House/Workshop",
        confidence=1,
        polarity=1,
    )


WORKSHOP_TOOLS: list[MeaningCentricStar] = [
    _tool_object(
        star_id="tool_obj_hammer",
        title_en="Hammer",
        title_pt="Martelo",
        title_ja="ハンマー",
        meaning_rpn="TOOL_OBJECT HAMMER STRIKE BUILD BASIC_OPERATIONS",
        visual_rpn=(
            "0.025 0.22 8 1 GEN_CYLINDER 0.0 0.11 0.0 MAT4_TRANSLATE MAT4_APPLY "
            "1.0 GEN_CUBE 0.16 0.08 0.08 MAT4_SCALE MAT4_APPLY 0.0 0.24 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
        ),
        taxonomy_refs=["tool_mathcore_tier1_scalar_worker_worker_v1"],
        house_position=(-0.60, 1.05, -0.18),
    ),
    _tool_object(
        star_id="tool_obj_wrench",
        title_en="Wrench",
        title_pt="Chave",
        title_ja="レンチ",
        meaning_rpn="TOOL_OBJECT WRENCH ALIGN GEOMETRY FIT",
        visual_rpn=(
            "1.0 GEN_CUBE 0.18 0.03 0.03 MAT4_SCALE MAT4_APPLY "
            "1.0 GEN_CUBE 0.03 0.12 0.03 MAT4_SCALE MAT4_APPLY 0.075 0.045 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "1.0 GEN_CUBE 0.06 0.03 0.03 MAT4_SCALE MAT4_APPLY 0.12 0.09 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
        ),
        taxonomy_refs=["tool_geom_profile_prep_v1", "tool_geom_bbox_crop_v1"],
        house_position=(-0.30, 1.05, -0.16),
    ),
    _tool_object(
        star_id="tool_obj_brush",
        title_en="Brush",
        title_pt="Pincel",
        title_ja="ブラシ",
        meaning_rpn="TOOL_OBJECT BRUSH PAINT BLEND SURFACE",
        visual_rpn=(
            "0.018 0.20 8 1 GEN_CYLINDER 0.0 0.10 0.0 MAT4_TRANSLATE MAT4_APPLY "
            "0.040 0.09 8 GEN_CONE 0.0 0.245 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
        ),
        taxonomy_refs=["tool_paint_gradient_backdrop_v1", "tool_paint_filter_stack_v1"],
        house_position=(0.00, 1.05, -0.14),
    ),
    _tool_object(
        star_id="tool_obj_tuning_fork",
        title_en="Tuning Fork",
        title_pt="Diapasao",
        title_ja="音叉",
        meaning_rpn="TOOL_OBJECT TUNING_FORK RESONATE AUDIO SIGNAL",
        visual_rpn=(
            "0.012 0.18 8 1 GEN_CYLINDER 0.0 0.09 0.0 MAT4_TRANSLATE MAT4_APPLY "
            "0.012 0.10 8 1 GEN_CYLINDER -0.03 0.23 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "0.012 0.10 8 1 GEN_CYLINDER 0.03 0.23 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "1.0 GEN_CUBE 0.07 0.02 0.02 MAT4_SCALE MAT4_APPLY 0.0 0.18 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
        ),
        taxonomy_refs=["tool_signal_audio_spectrogram_v1", "tool_codec_audio_mdct_v1"],
        house_position=(0.30, 1.05, -0.18),
    ),
    _tool_object(
        star_id="tool_obj_lens",
        title_en="Lens",
        title_pt="Lente",
        title_ja="レンズ",
        meaning_rpn="TOOL_OBJECT LENS INSPECT CODEC DETAIL",
        visual_rpn=(
            "0.055 0.014 16 8 GEN_TORUS "
            "1.0 GEN_CUBE 0.10 0.01 0.02 MAT4_SCALE MAT4_APPLY 0.07 -0.03 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
        ),
        taxonomy_refs=["tool_codec_ternary_blocks_v1", "tool_codec_video_dct8_grid_v1"],
        house_position=(0.60, 1.05, -0.14),
    ),
]


__all__ = ["WORKSHOP_TOOLS"]
