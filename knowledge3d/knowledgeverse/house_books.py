"""Book-shaped House objects that act as loadable galaxy containers."""

from __future__ import annotations

from ._house_utils import surface_forms
from .meaning_star import MeaningCentricStar


def _book_visual_rpn(width: float, height: float, depth: float) -> str:
    pages_w = width * 0.94
    pages_h = height * 0.96
    pages_d = depth * 0.84
    cover_w = width
    cover_h = height
    cover_d = max(0.003, depth * 0.08)
    cover_offset = (pages_d / 2.0) + (cover_d / 2.0)
    spine_w = max(0.003, width * 0.06)
    return (
        f"1.0 GEN_CUBE {pages_w:.4f} {pages_h:.4f} {pages_d:.4f} MAT4_SCALE MAT4_APPLY "
        f"1.0 GEN_CUBE {cover_w:.4f} {cover_h:.4f} {cover_d:.4f} MAT4_SCALE MAT4_APPLY "
        f"0.0 0.0 {cover_offset:.4f} MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
        f"1.0 GEN_CUBE {cover_w:.4f} {cover_h:.4f} {cover_d:.4f} MAT4_SCALE MAT4_APPLY "
        f"0.0 0.0 {-cover_offset:.4f} MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
        f"1.0 GEN_CUBE {spine_w:.4f} {cover_h:.4f} {depth:.4f} MAT4_SCALE MAT4_APPLY "
        f"{(-cover_w / 2.0):.4f} 0.0 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
    )


def _book(
    *,
    star_id: str,
    title_en: str,
    title_pt: str,
    title_ja: str,
    galaxy_ref: str,
    meaning_rpn: str,
    taxonomy_refs: list[str],
    width: float,
    height: float,
    depth: float,
    house_position: tuple[float, float, float],
) -> MeaningCentricStar:
    return MeaningCentricStar(
        star_id=star_id,
        meaning_class="book",
        meaning_rpn=meaning_rpn,
        domain="House/Library/Books",
        taxonomy_refs=taxonomy_refs,
        surface_forms=surface_forms(title_en, title_pt, title_ja),
        visual_rpn=_book_visual_rpn(width, height, depth),
        behavior_rpn="BOOK OPEN LOAD_GALAXY READ_SEQUENCE",
        house_position=house_position,
        house_room="House/Library",
        galaxy_ref=galaxy_ref,
        confidence=1,
        polarity=1,
    )


HOUSE_BOOKS: list[MeaningCentricStar] = [
    _book(
        star_id="book_mathematics_primer",
        title_en="Mathematics Primer",
        title_pt="Primer de Matematica",
        title_ja="数学入門",
        galaxy_ref="Book/MathematicsPrimer",
        meaning_rpn="BOOK MATHEMATICS PRIMER NUMBERS OPERATIONS PATTERNS LOAD_GALAXY",
        taxonomy_refs=["concept_mathematics", "num_0", "num_1", "num_2"],
        width=0.19,
        height=0.27,
        depth=0.036,
        house_position=(-1.80, 0.15, -0.12),
    ),
    _book(
        star_id="book_language_foundations",
        title_en="Language Foundations",
        title_pt="Fundamentos da Linguagem",
        title_ja="言語の基礎",
        galaxy_ref="Book/LanguageFoundations",
        meaning_rpn="BOOK LANGUAGE FOUNDATIONS CHARACTERS WORDS GRAMMAR LOAD_GALAXY",
        taxonomy_refs=["concept_language", "word_zero", "word_one", "word_two"],
        width=0.18,
        height=0.265,
        depth=0.032,
        house_position=(-1.58, 0.15, -0.12),
    ),
    _book(
        star_id="book_physics_handbook",
        title_en="Physics Handbook",
        title_pt="Manual de Fisica",
        title_ja="物理ハンドブック",
        galaxy_ref="Book/PhysicsHandbook",
        meaning_rpn="BOOK PHYSICS HANDBOOK FORCE MOTION FIELD LOAD_GALAXY",
        taxonomy_refs=["concept_physics", "reality_anchor_college_physics_core", "reality_law_newton_second"],
        width=0.205,
        height=0.275,
        depth=0.040,
        house_position=(-1.34, 0.15, -0.12),
    ),
    _book(
        star_id="book_biology_atlas",
        title_en="Biology Atlas",
        title_pt="Atlas de Biologia",
        title_ja="生物学アトラス",
        galaxy_ref="Book/BiologyAtlas",
        meaning_rpn="BOOK BIOLOGY ATLAS CELL ORGANISM ECOLOGY LOAD_GALAXY",
        taxonomy_refs=["concept_biology", "reality_anchor_college_biology_core", "reality_biology_cell_theory"],
        width=0.175,
        height=0.255,
        depth=0.030,
        house_position=(-1.11, 0.15, -0.12),
    ),
    _book(
        star_id="book_tool_manual",
        title_en="Tool Manual",
        title_pt="Manual de Ferramentas",
        title_ja="道具マニュアル",
        galaxy_ref="Book/ToolManual",
        meaning_rpn="BOOK TOOL MANUAL CONSTRUCTION METHODS APPLY LOAD_GALAXY",
        taxonomy_refs=["concept_tool", "rotate_90", "obj3d_gen_cube"],
        width=0.165,
        height=0.248,
        depth=0.028,
        house_position=(-0.90, 0.15, -0.12),
    ),
]


__all__ = ["HOUSE_BOOKS"]
