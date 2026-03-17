from __future__ import annotations

from pathlib import Path

from knowledge3d.cranium.bridges.mesh_bridge import MeshBridge
from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager
from knowledge3d.knowledgeverse.house_builder import build_house
from knowledge3d.knowledgeverse.house_books import HOUSE_BOOKS
from knowledge3d.knowledgeverse.house_doors import HOUSE_DOORS
from knowledge3d.knowledgeverse.house_furniture import HOUSE_FURNITURE
from knowledge3d.knowledgeverse.house_gallery_displays import GALLERY_DISPLAYS
from knowledge3d.knowledgeverse.house_knowledge_tree import KNOWLEDGE_TREE_BRANCHES
from knowledge3d.knowledgeverse.house_memory_tablet import MEMORY_TABLET
from knowledge3d.knowledgeverse.house_observatory import OBSERVATORY_INSTRUMENTS
from knowledge3d.knowledgeverse.house_rooms import HOUSE_ROOMS
from knowledge3d.knowledgeverse.house_workshop_tools import WORKSHOP_TOOLS
from knowledge3d.knowledgeverse.seed_stars import SEED_STARS


def test_house_templates_have_multilingual_surface_forms():
    for star in (
        HOUSE_ROOMS
        + HOUSE_FURNITURE
        + HOUSE_DOORS
        + WORKSHOP_TOOLS
        + HOUSE_BOOKS
        + GALLERY_DISPLAYS
        + OBSERVATORY_INSTRUMENTS
        + [MEMORY_TABLET]
        + KNOWLEDGE_TREE_BRANCHES
    ):
        assert star.visual_rpn
        assert set(star.surface_forms) == {"en", "pt", "ja"}
        for surface_form in star.surface_forms.values():
            assert surface_form.word_ref
            assert surface_form.char_refs


def test_house_template_visual_programs_execute() -> None:
    bridge = MeshBridge()
    for star in (
        HOUSE_ROOMS
        + HOUSE_FURNITURE
        + HOUSE_DOORS
        + WORKSHOP_TOOLS
        + HOUSE_BOOKS
        + GALLERY_DISPLAYS
        + OBSERVATORY_INSTRUMENTS
        + [MEMORY_TABLET]
        + KNOWLEDGE_TREE_BRANCHES
    ):
        result = bridge.execute_rpn_program(star.visual_rpn or "")
        assert result.mesh.vertices
        assert result.mesh.triangles
        assert result.mesh.normals


def test_build_house_stores_templates_and_seeds(tmp_path: Path) -> None:
    manager = GalaxyManager(storage_root=tmp_path / "galaxies")
    summary = build_house(manager)
    house = manager.get_galaxy("House")
    assert summary["rooms"] == len(HOUSE_ROOMS)
    assert summary["furniture"] == len(HOUSE_FURNITURE)
    assert summary["doors"] == len(HOUSE_DOORS)
    assert summary["tools"] == len(WORKSHOP_TOOLS)
    assert summary["books"] == len(HOUSE_BOOKS)
    assert summary["displays"] == len(GALLERY_DISPLAYS)
    assert summary["instruments"] == len(OBSERVATORY_INSTRUMENTS)
    assert summary["tablet"] == 1
    assert summary["tree_nodes"] == len(KNOWLEDGE_TREE_BRANCHES)
    assert summary["seed_stars"] == len(SEED_STARS)
    assert len(house.entries) == (
        len(HOUSE_ROOMS)
        + len(HOUSE_FURNITURE)
        + len(HOUSE_DOORS)
        + len(WORKSHOP_TOOLS)
        + len(HOUSE_BOOKS)
        + len(GALLERY_DISPLAYS)
        + len(OBSERVATORY_INSTRUMENTS)
        + 1
        + len(KNOWLEDGE_TREE_BRANCHES)
        + len(SEED_STARS)
    )
    assert manager.load_meaning_star("House", "room_library") is not None
    assert manager.load_meaning_star("House", "furniture_bookshelf") is not None
    assert manager.load_meaning_star("House", "door_library_garden") is not None
    assert manager.load_meaning_star("House", "tool_obj_hammer") is not None
    assert manager.load_meaning_star("House", "book_mathematics_primer") is not None
    assert manager.load_meaning_star("House", "display_number_line") is not None
    assert manager.load_meaning_star("House", "observatory_telescope") is not None
    assert manager.load_meaning_star("House", MEMORY_TABLET.star_id) is not None
    assert manager.load_meaning_star("House", "tree_branch_mathematics") is not None
    assert manager.load_meaning_star("House", SEED_STARS[0].star_id) is not None
    assert summary["room_meshes"]["room_library"].mesh.vertices
    assert summary["furniture_meshes"]["furniture_bookshelf"].mesh.triangles
    assert summary["door_meshes"]["door_library_garden"].mesh.triangles
    assert summary["tool_meshes"]["tool_obj_hammer"].mesh.triangles
    assert summary["book_meshes"]["book_mathematics_primer"].mesh.triangles
    assert summary["display_meshes"]["display_number_line"].mesh.triangles
    assert summary["instrument_meshes"]["observatory_telescope"].mesh.triangles
    assert summary["tablet_meshes"][MEMORY_TABLET.star_id].mesh.triangles
    assert summary["tree_meshes"]["tree_branch_mathematics"].mesh.triangles
