from __future__ import annotations

from knowledge3d.knowledgeverse.house_books import HOUSE_BOOKS
from knowledge3d.knowledgeverse.house_doors import HOUSE_DOORS
from knowledge3d.knowledgeverse.house_furniture import HOUSE_FURNITURE
from knowledge3d.knowledgeverse.house_gallery_displays import GALLERY_DISPLAYS
from knowledge3d.knowledgeverse.house_knowledge_tree import KNOWLEDGE_TREE_BRANCHES
from knowledge3d.knowledgeverse.house_memory_tablet import MEMORY_TABLET
from knowledge3d.knowledgeverse.house_observatory import OBSERVATORY_INSTRUMENTS
from knowledge3d.knowledgeverse.house_rooms import HOUSE_ROOMS
from knowledge3d.knowledgeverse.house_workshop_tools import WORKSHOP_TOOLS


def test_house_visual_rpn_programs_are_parseable() -> None:
    known_ops = {
        "GEN_CUBE",
        "GEN_CYLINDER",
        "GEN_CONE",
        "GEN_TORUS",
        "GEN_UV_SPHERE",
        "GEN_PLANE",
        "GEN_ICOSPHERE",
        "MAT4_IDENTITY",
        "MAT4_TRANSLATE",
        "MAT4_SCALE",
        "MAT4_ROTATE_X",
        "MAT4_ROTATE_Y",
        "MAT4_ROTATE_Z",
        "MAT4_MUL",
        "MAT4_APPLY",
        "CSG_UNION",
        "CSG_SUBTRACT",
        "CSG_INTERSECT",
        "MOVE",
        "LINE",
        "QUAD",
        "CUBIC",
        "ARC",
        "CLOSE",
        "EXTRUDE",
        "LATHE",
    }
    stars = (
        HOUSE_ROOMS
        + HOUSE_FURNITURE
        + HOUSE_DOORS
        + WORKSHOP_TOOLS
        + HOUSE_BOOKS
        + GALLERY_DISPLAYS
        + OBSERVATORY_INSTRUMENTS
        + KNOWLEDGE_TREE_BRANCHES
        + [MEMORY_TABLET]
    )
    for star in stars:
        if not star.visual_rpn:
            continue
        for token in star.visual_rpn.split():
            try:
                float(token)
            except ValueError:
                assert token in known_ops, f"Unknown op '{token}' in {star.star_id}"
