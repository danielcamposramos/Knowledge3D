"""Door stars connecting adjacent House rooms."""

from __future__ import annotations

import math

from ._house_utils import surface_forms
from .meaning_star import MeaningCentricStar


def _door_frame_visual(rotation_y: float = 0.0) -> str:
    visual = (
        "1.0 GEN_CUBE 1.40 2.20 0.20 MAT4_SCALE MAT4_APPLY "
        "1.0 GEN_CUBE 1.00 1.80 0.24 MAT4_SCALE MAT4_APPLY "
        "0.0 -0.10 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_SUBTRACT"
    )
    if rotation_y:
        visual += f" {rotation_y:.4f} MAT4_ROTATE_Y MAT4_APPLY"
    return visual


def _door(
    *,
    star_id: str,
    title_en: str,
    title_pt: str,
    title_ja: str,
    room_a: str,
    room_b: str,
    house_position: tuple[float, float, float],
    rotation_y: float = 0.0,
) -> MeaningCentricStar:
    return MeaningCentricStar(
        star_id=star_id,
        meaning_class="door",
        meaning_rpn=f"DOOR CONNECT {room_a.split('/')[-1].upper()} {room_b.split('/')[-1].upper()}",
        domain="House/Connectivity",
        taxonomy_refs=[room_a, room_b],
        surface_forms=surface_forms(title_en, title_pt, title_ja),
        visual_rpn=_door_frame_visual(rotation_y),
        behavior_rpn=f"DOOR_TRAVERSE CONNECT {room_a} {room_b}",
        house_position=house_position,
        house_room=room_a,
        confidence=1,
        polarity=1,
    )


HOUSE_DOORS: list[MeaningCentricStar] = [
    _door(
        star_id="door_library_garden",
        title_en="Library Garden Door",
        title_pt="Porta Biblioteca Jardim",
        title_ja="図書館と庭の扉",
        room_a="House/Library",
        room_b="House/Garden",
        house_position=(9.0, 0.0, 0.0),
        rotation_y=math.pi / 2.0,
    ),
    _door(
        star_id="door_garden_workshop",
        title_en="Garden Workshop Door",
        title_pt="Porta Jardim Oficina",
        title_ja="庭と工房の扉",
        room_a="House/Garden",
        room_b="House/Workshop",
        house_position=(9.0, 0.0, 0.0),
        rotation_y=math.pi / 2.0,
    ),
    _door(
        star_id="door_workshop_gallery",
        title_en="Workshop Gallery Door",
        title_pt="Porta Oficina Galeria",
        title_ja="工房とギャラリーの扉",
        room_a="House/Workshop",
        room_b="House/Gallery",
        house_position=(9.0, 0.0, 0.0),
        rotation_y=math.pi / 2.0,
    ),
    _door(
        star_id="door_gallery_bathtub",
        title_en="Gallery Observatory Door",
        title_pt="Porta Galeria Observatorio",
        title_ja="ギャラリーと観測室の扉",
        room_a="House/Gallery",
        room_b="House/Bathtub",
        house_position=(9.0, 0.0, 0.0),
        rotation_y=math.pi / 2.0,
    ),
]


__all__ = ["HOUSE_DOORS"]
