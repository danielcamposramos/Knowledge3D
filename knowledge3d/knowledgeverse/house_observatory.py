"""Observatory instruments for the Bathtub introspection room."""

from __future__ import annotations

from ._house_utils import surface_forms
from .meaning_star import MeaningCentricStar


def _instrument(
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
        meaning_class="instrument",
        meaning_rpn=meaning_rpn,
        domain="House/Bathtub/Observatory",
        taxonomy_refs=taxonomy_refs,
        surface_forms=surface_forms(title_en, title_pt, title_ja),
        visual_rpn=visual_rpn,
        behavior_rpn="OBSERVE INTROSPECT RECORD_GALAXY_VIEW",
        house_position=house_position,
        house_room="House/Bathtub",
        confidence=1,
        polarity=1,
    )


OBSERVATORY_INSTRUMENTS: list[MeaningCentricStar] = [
    _instrument(
        star_id="observatory_telescope",
        title_en="Telescope",
        title_pt="Telescopio",
        title_ja="望遠鏡",
        meaning_rpn="INSTRUMENT TELESCOPE OBSERVE GALAXY OUTWARD",
        visual_rpn=(
            "0.06 0.50 12 1 GEN_CYLINDER 0.0 0.25 0.0 MAT4_TRANSLATE MAT4_APPLY "
            "0.09 0.12 12 GEN_CONE 0.0 0.56 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
            "-0.35 MAT4_ROTATE_X MAT4_APPLY"
        ),
        taxonomy_refs=["concept_self_reflection", "concept_physics", "concept_growth"],
        house_position=(0.0, 0.8, -2.0),
    ),
    _instrument(
        star_id="observatory_prism",
        title_en="Prism",
        title_pt="Prisma",
        title_ja="プリズム",
        meaning_rpn="INSTRUMENT PRISM SPLIT SIGNAL INTO SPECTRA",
        visual_rpn=(
            "0.0 0.0 MOVE 0.16 0.0 LINE 0.08 0.14 LINE CLOSE 0.16 EXTRUDE "
            "1.5708 MAT4_ROTATE_X MAT4_APPLY 0.7854 MAT4_ROTATE_Z MAT4_APPLY"
        ),
        taxonomy_refs=["concept_visual_art", "concept_physics", "concept_growth"],
        house_position=(1.8, 0.3, -1.0),
    ),
    _instrument(
        star_id="observatory_journal",
        title_en="Journal",
        title_pt="Diario",
        title_ja="観測日誌",
        meaning_rpn="INSTRUMENT JOURNAL RECORD OBSERVATION REFLECTION",
        visual_rpn=(
            "1.0 GEN_CUBE 0.18 0.01 0.14 MAT4_SCALE MAT4_APPLY "
            "-0.10 0.0 0.0 MAT4_TRANSLATE MAT4_APPLY 0.18 MAT4_ROTATE_Z MAT4_APPLY "
            "1.0 GEN_CUBE 0.18 0.01 0.14 MAT4_SCALE MAT4_APPLY "
            "0.10 0.0 0.0 MAT4_TRANSLATE MAT4_APPLY -0.18 MAT4_ROTATE_Z MAT4_APPLY CSG_UNION "
            "1.0 GEN_CUBE 0.02 0.015 0.15 MAT4_SCALE MAT4_APPLY CSG_UNION"
        ),
        taxonomy_refs=["concept_self_reflection", "concept_language", "concept_growth"],
        house_position=(-1.5, 0.3, 0.5),
    ),
]


__all__ = ["OBSERVATORY_INSTRUMENTS"]
